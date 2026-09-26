"""NF4WeightStore: QLoRA-style blockwise NF4 quantization + double
quantization of the per-block scale factors.
See docs/design/04-lora-adapter-mechanics-and-loss-weighting.md section 3.3 for design rationale.

**The codebook is grounded directly in bitsandbytes' real, current
source** (bitsandbytes/functional.py's create_normal_map(), fetched and
read directly), reproduced here in pure PyTorch (torch.special.ndtri,
the inverse standard-normal CDF, in place of scipy.stats.norm.ppf, to
avoid a new scipy dependency) rather than hand-derived from the paper's
own equations, which describe the *method* (quantile estimation of a
standard normal) but aren't runnable code. Cross-checked against
bitsandbytes' own published codebook constants directly -- matches to
~1e-7, the expected float32 rounding difference between two
independently-computed inverse-CDF evaluations, not a real discrepancy
(see smoke_test_nf4_weight_store.py).

**Block-wise quantization** (blocksize=64, bitsandbytes' own default for
4-bit types): each contiguous 64-element block of the flattened weight
gets its own absmax scale; every element is normalized by that scale,
then mapped to the nearest of NF4's 16 codebook values, stored as a
4-bit index -- two indices packed per uint8 byte (storing one index per
*byte* would be real 8-bit storage, not 4-bit, defeating the entire
point of this class).

**Double quantization, real but deliberately simplified from
bitsandbytes' own second level.** Confirmed directly from
bitsandbytes' quantize_4bit(compress_statistics=True): the per-block
absmax values themselves get centered (subtract their mean) and
quantized again, in blocks of 256, via bitsandbytes' general-purpose
"dynamic" 8-bit map (create_dynamic_map() -- a full floating-point-like
signed 8-bit scheme, itself a separate, involved piece of machinery).
This module quantizes the centered absmax values with plain linear
min-max 8-bit quantization instead -- a real, honest simplification, not
a byte-exact reproduction of bitsandbytes' own second level. The savings
this buys (~0.37 bits/parameter in the QLoRA paper, roughly 9% off an
already-small ~0.5-bit/parameter absmax overhead) is real either way;
matching bitsandbytes' specific dynamic-map scheme bit-for-bit would add
real implementation complexity for a small share of this class's total
value, nearly all of which comes from the primary 4-bit quantization
above.

**Wired into a real forward pass.** materialize() exists specifically so
a real AdapterStrategy could call it each forward to get a fresh
dequantized tensor -- PlainLoRAAdapter now does exactly that: it honors
both BF16WeightStore and NF4WeightStore, checked at wrap() time (see
adapter_strategy.py's module docstring), constructing
nodes/model/nf4_lora_layer.py's NF4LoRALinear/NF4LoRAConv2d for the
NF4 case, which call frozen.materialize() every forward. DoRAAdapter
still only honors BF16WeightStore -- see adapter_strategy.py's
DoRAAdapter docstring for why not. This module remains the storage/
(de)quantization piece on its own, tested against round-trip
reconstruction error, not against a trained model's real quality (the
diffusion-specific caveat in design doc section 3.3 needs an actual
training run to check, not possible in this environment).

**materialize()'s caching decision, made explicitly rather than left
open** (design doc's own words: "a real implementation needs a genuine
decision about caching ... a real VRAM/speed tradeoff this design
doesn't resolve for you"): re-dequantize on every call, never cache the
dequantized bf16 result across calls. Caching would mean holding a full
bf16 copy of the weight resident between forward passes -- exactly the
VRAM cost NF4 storage exists to avoid, for however long the cache lives.
A future MemoryManager-backed scratch buffer changes *where* the
dequantized tensor's memory comes from (a pooled, reused buffer instead
of a fresh allocation each call), not *whether* it's recomputed each
call -- recomputing stays correct either way.
"""

from __future__ import annotations

import torch

from .frozen_weight_store import FrozenWeightStore
from ..quantization import dequantize_blockwise_linear_u8, quantize_blockwise_linear_u8

_NF4_CODEBOOK_CACHE: dict = {}


def _nf4_codebook(device, dtype=torch.float32) -> torch.Tensor:
    """16 quantization levels, grounded directly in bitsandbytes'
    create_normal_map() (offset=0.9677083, use_extra_value=True -- its
    own defaults) -- see this module's docstring. torch.special.ndtri in
    place of scipy.stats.norm.ppf (the same function, inverse
    standard-normal CDF); computed in float64 for precision then cast
    down, matching how a one-time, 16-value constant should be computed.
    Cached per (device, dtype) -- small and fixed, no reason to recompute
    identically on every call."""
    key = (device, dtype)
    if key not in _NF4_CODEBOOK_CACHE:
        offset = 0.9677083
        v1 = torch.special.ndtri(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
        v3 = -torch.special.ndtri(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
        values = torch.cat([v1, torch.zeros(1, dtype=torch.float64), v3])
        values = values.sort().values
        values = values / values.max()
        _NF4_CODEBOOK_CACHE[key] = values.to(device=device, dtype=dtype)
    return _NF4_CODEBOOK_CACHE[key]


def _quantize_blockwise_nearest(flat: torch.Tensor, codebook: torch.Tensor,
                                 blocksize: int) -> tuple[torch.Tensor, torch.Tensor]:
    """flat: 1D. Pads to a whole number of blocksize-sized blocks (NOT
    trimmed back here -- the caller decides when trimming to the real
    element count matters, so packing/unpacking always operates on a
    consistent, block-aligned length). Returns (indices: uint8, length ==
    num_blocks*blocksize; absmax: float32, length == num_blocks)."""
    n = flat.numel()
    num_blocks = -(-n // blocksize)  # ceil division
    padded_n = num_blocks * blocksize
    padded = flat.new_zeros(padded_n)
    padded[:n] = flat
    blocks = padded.view(num_blocks, blocksize)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    normalized = blocks / absmax.unsqueeze(1)
    # Nearest of 16 codebook values -- brute-force distance. Cheap: only
    # 16 candidates per element, not a real bottleneck next to the
    # quantization this buys.
    dist = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs()
    indices = dist.argmin(dim=-1).to(torch.uint8).view(-1)
    return indices, absmax.to(torch.float32)


def _pack_nibbles(indices: torch.Tensor) -> torch.Tensor:
    """Two 4-bit indices per byte. indices is expected to already have an
    even length in this module's real usage (blocksize is always even),
    but pads with one zero index if not, for general correctness."""
    n = indices.numel()
    if n % 2 == 1:
        indices = torch.cat([indices, torch.zeros(1, dtype=torch.uint8, device=indices.device)])
    pairs = indices.view(-1, 2)
    return (pairs[:, 0] | (pairs[:, 1] << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of _pack_nibbles -- n is the real number of indices to
    return (always block-aligned in this module's own usage, not the
    original unpadded element count -- see _quantize_blockwise_nearest's
    docstring)."""
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    interleaved = torch.stack([low, high], dim=1).view(-1)
    return interleaved[:n]



class NF4WeightStore(FrozenWeightStore):
    """See this module's docstring for the full quantization scheme."""

    def __init__(self, weight: torch.Tensor, blocksize: int = 64,
                 double_quant_blocksize: int = 256):
        if blocksize % 2 != 0:
            raise ValueError(f"blocksize must be even (nibble-packing needs it), got {blocksize}")
        self._shape = weight.shape
        self._numel = weight.numel()
        self._blocksize = blocksize
        # bf16 unless the source was fp16 -- matches this project's own
        # working precision (see core/lora.py's LoRALinear) rather than
        # assuming bf16 always.
        self._dtype = weight.dtype if weight.dtype == torch.float16 else torch.bfloat16

        flat = weight.detach().to(torch.float32).reshape(-1)
        codebook = _nf4_codebook(weight.device, torch.float32)
        indices, absmax = _quantize_blockwise_nearest(flat, codebook, blocksize)
        self._packed = _pack_nibbles(indices)  # indices already block-aligned/even
        self._num_blocks = absmax.numel()

        # Double quantization: center the per-block absmax values, then
        # quantize *them* with a simple linear 8-bit scheme, in their own
        # (larger) blocks -- see this module's docstring. Shared
        # nodes/quantization.py utility now, not a private per-block loop --
        # same math, vectorized across all double-quant blocks at once.
        self._dq_blocksize = double_quant_blocksize
        self._offset = absmax.mean()
        centered = absmax - self._offset
        self._absmax_q, self._absmax_lo, self._absmax_scale = quantize_blockwise_linear_u8(
            centered, double_quant_blocksize)

    def footprint_bytes(self) -> int:
        """Real compressed size: packed 4-bit codes + double-quantized
        absmax (uint8 codes + one fp32 lo/scale pair per double-quant
        block, negligible next to the codes themselves) + the one scalar
        offset. Deliberately does NOT count the transient dequantized
        tensor materialize() allocates -- that's not storage this class
        holds, it's a fresh, temporary tensor the caller owns for one
        forward pass (see this module's docstring on why nothing is
        cached)."""
        return (
            self._packed.numel() * self._packed.element_size()
            + self._absmax_q.numel() * self._absmax_q.element_size()
            + self._absmax_lo.numel() * self._absmax_lo.element_size()
            + self._absmax_scale.numel() * self._absmax_scale.element_size()
            + self._offset.numel() * self._offset.element_size()
        )

    @property
    def dtype(self) -> torch.dtype:
        """materialize()'s own output dtype, without paying for a full
        dequantize just to read it -- nodes/model/nf4_lora_layer.py's
        NF4LoRALinear/NF4LoRAConv2d both used to call
        frozen.materialize().dtype during __init__ purely to size
        base_bias's cast, materializing (dequantizing every block,
        allocating a full-precision copy of the whole weight) and then
        immediately throwing the result away -- a real, if one-time,
        wasted allocation and compute pass per frozen layer at model-
        construction time. self._dtype already holds exactly this."""
        return self._dtype

    def materialize(self) -> torch.Tensor:
        """Re-dequantizes fresh every call -- see this module's docstring
        for why nothing is cached."""
        absmax = dequantize_blockwise_linear_u8(
            self._absmax_q, self._absmax_lo, self._absmax_scale, self._num_blocks)
        absmax = absmax + self._offset

        codebook = _nf4_codebook(self._packed.device, torch.float32)
        padded_n = self._num_blocks * self._blocksize
        indices = _unpack_nibbles(self._packed, padded_n)
        codes = codebook[indices.long()]
        values = (codes.view(self._num_blocks, self._blocksize)
                  * absmax.unsqueeze(1)).view(-1)[:self._numel]
        return values.view(self._shape).to(self._dtype)
