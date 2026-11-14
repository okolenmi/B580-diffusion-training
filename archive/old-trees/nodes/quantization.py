"""Shared block-wise linear 8-bit quantization -- the exact scheme
nodes/model/nf4_weight_store.py already used privately for its own
double-quantization of per-block absmax values (see that module's own
docstring for the full "why linear, not bitsandbytes' own non-linear
dynamic map" reasoning, unchanged here), extracted so
nodes/optimizer/state_store.py's optimizer-state quantization reuses
the exact same math rather than a second, subtly-different
implementation of it.

blocksize=256 (both call sites' own default) is bitsandbytes' own real,
current default for exactly this kind of block-wise 8-bit quantization
-- checked directly against their source, not assumed: the modern
optimizer_update_8bit_blockwise() "uses per-block absmax arrays (block
size 256) for much better numerical accuracy" than the older,
now-removed single-global-scale path.

Vectorized across all blocks at once (blocks.min(dim=1)/.max(dim=1)),
not a Python loop over blocks -- nf4_weight_store.py's own original
version looped one block at a time, fine for double-quantizing a
handful of per-block absmax values, too slow for quantizing a full
optimizer-state tensor with potentially millions of elements.
"""

from __future__ import annotations

import torch


def quantize_blockwise_linear_u8(flat: torch.Tensor, blocksize: int
                                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """flat: 1D. Pads to a whole number of blocksize-sized blocks (not
    trimmed back here -- dequantize_blockwise_linear_u8's own `n`
    argument is where trimming to the real element count happens, so
    the padded/quantized representation itself always has a consistent,
    block-aligned length). Returns (q: uint8 [num_blocks, blocksize],
    lo: float32 [num_blocks], scale: float32 [num_blocks])."""
    n = flat.numel()
    num_blocks = -(-n // blocksize)  # ceil division
    padded_n = num_blocks * blocksize
    padded = flat.new_zeros(padded_n)
    padded[:n] = flat
    blocks = padded.view(num_blocks, blocksize)
    lo = blocks.min(dim=1).values
    hi = blocks.max(dim=1).values
    scale = (hi - lo).clamp_min(1e-12) / 255.0
    q = ((blocks - lo.unsqueeze(1)) / scale.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8)
    return q, lo.to(torch.float32), scale.to(torch.float32)


def dequantize_blockwise_linear_u8(q: torch.Tensor, lo: torch.Tensor, scale: torch.Tensor,
                                    n: int) -> torch.Tensor:
    """Inverse of quantize_blockwise_linear_u8 above. n is the real,
    unpadded element count to return -- the caller's own concern, not
    recoverable from q/lo/scale alone (they only know the padded,
    block-aligned length)."""
    values = (q.to(torch.float32) * scale.unsqueeze(1) + lo.unsqueeze(1)).view(-1)
    return values[:n]
