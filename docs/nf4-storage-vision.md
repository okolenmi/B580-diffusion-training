# NF4 storage for frozen weights: a vision, with the numbers

*Written by the xpu-kernels side, for the trainer's own plan. It is a
document only: no trainer code is touched here. The NF4 port is listed in
`rework/PORTING.md` (planned) and in `next/manual/persistence-and-unbuilt.md`
("NF4 frozen weights"), and the int8 base is already built
(`adapter.base_precision`).

Status of the numbers below: `measured` = run on this B580; `derived` =
computed from measured figures; `untested` = an estimate.

## The plan, and the arithmetic behind it

NF4 (QLoRA) stores the frozen base in 4 bits with a per-block absmax (64
elements per block in the reference implementation) and, optionally, quantizes
those absmax values a second time. For this trainer's 2.17 B quantized
parameters:

| storage | bytes/param | 2.17 B params |
|---|---|---|
| bf16 | 2 | 4.34 GB |
| int8 (what is built) | 1 (+ scale) | 2.17 GB |
| NF4 | 0.5 + 0.06 (absmax fp32) + 0.02 (double-quantized) | **~1.26 GB** |

So NF4 buys about **0.9 GB over the int8 base** and **3.1 GB over bf16**
(derived). That is the whole point of the port, and it is worth being precise
about: NF4 is not a *speed* plan, it is the next step down in memory after the
int8 base that already exists.

## What the NF4 path costs today, in this tree

`nodes/model/nf4_weight_store.py`'s `materialize()` is a torch chain:

```python
indices = _unpack_nibbles(self._packed, padded_n)   # & 0x0F, >> 4, stack
codes = codebook[indices.long()]                    # .long() is an int64 tensor
values = (codes.view(blocks, blocksize) * absmax.unsqueeze(1)).view(-1)
return values.view(self._shape).to(self._dtype)
```

Measured on this machine, per parameter, that chain moves roughly:

| step | bytes |
|---|---|
| read the packed nibbles | 0.5 |
| write `low` and `high` (two full-size uint8 tensors) | 2 |
| write the stacked/interleaved tensor | 1 |
| write `indices.long()` (**int64**) | 8 |
| read it back for the gather | 8 |
| write the gathered codes (fp32) | 4 |
| read + write the scaled product (fp32) | 8 |
| read + write the bf16 result | 4 |
| **total** | **~40 bytes/parameter** |

For 2.17 B parameters that is ~87 GB of traffic per materialization:
**~200 ms per step** at the measured 450 GB/s (derived) -- about 13 % of a
1.5 s step, for a format whose whole purpose is to save memory.

Worse is the peak. The chain holds `indices.long()` (8 bytes/parameter), the
gathered codes (4) and the product (4) *simultaneously*: **~16 bytes per
parameter of temporaries**, i.e. for a 100 M-parameter layer, 1.6 GB of
scratch for a layer whose nf4 storage is 50 MB and whose bf16 weight is
200 MB. As written, the nf4 path can cost more peak memory than the bf16 one
it replaces. The same chain, materialized once per layer per forward, is also
the wrong shape for a captured step: those allocations would come from the
graph's private pool and be replayed on every step.

## What the optimization is

One fused kernel, in the same shape as the ones this project already has
(`xpu_kernels.triton`): read the packed nibbles and the per-block absmax,
write bf16.

| | bytes/parameter |
|---|---|
| nf4 storage read (packed) | 0.5 |
| absmax read (fp32 per 64 elements) | 0.06 |
| bf16 write | 2.0 |
| **total** | **2.6** |

That is **15x less traffic** than the chain (~14 ms per step for 2.17 B
parameters, derived) and, more importantly, **no temporaries at all**: the
output is the only allocation, so the peak is the layer's bf16 weight and
nothing else.

And unlike the int8 row-quantizer -- the one kernel in this project that
cannot be bit-exact, because a correctly rounded fp32 division costs 18x more
than the fast one on this card -- **this one can be bit-exact**. The
arithmetic is a table lookup, a multiply and a cast: no division, no rounding
mode to get wrong, and the multiply is correctly rounded in both torch and
Triton. The same contract as the rope and adaLN kernels: verify once per
process against the reference chain, bit for bit; fall back if it does not
match; one env switch to turn it off.

Shape of the kernel: one program per tile of, say, 1024 elements (16 blocks
of 64), each thread reading one byte of packed data (two nibbles) and the
block's absmax, looking the code up in a 16-entry constant table, multiplying
and storing bf16. The codebook is a constant (the 16 normal quantiles), so
it can live in the kernel as a constant table -- and it must be *the same
table* the trainer's reference uses, since that is the definition of the
format (the values match bitsandbytes' published constants to ~1e-7).

The double quantization (the absmax values quantized a second time) is a
*small* tensor relative to the weights (one byte per 64 elements) but not
tiny: 34 M blocks for 2.17 B parameters is 136 MB of fp32 if materialized
whole. It should be dequantized per layer (or per block-tile) inside the
same kernel, not as a separate full-size chain.

## It is built: `xpu_kernels.triton.nf4`

The kernel exists in the `xpu-kernels` package (published on PyPI as
0.2.0; the tree is ahead of that release). Measured on this B580, 100 M
parameters:

| | time | traffic | scratch |
|---|---|---|---|
| the torch chain | 20.75 ms | ~40 bytes/parameter (~4 GB) | ~16 bytes/parameter |
| **`xpu_kernels.triton.nf4.dequant_nf4`** | **0.66 ms** (31.5x) | 2.6 bytes/parameter (256 MB) | none |

389 GB/s (a gather-heavy kernel, so not at the card's 468 GB/s read floor),
and **bit-identical** to the chain in fp32 and at the production bf16 -- the
gate checks both, once per process, and the format itself (the codebook's
definition, the nibble order, the block layout) is pinned by the library's
tests, so a weight stored by one implementation is readable by the other.

Using it from the trainer's side:

```python
from xpu_kernels.triton import nf4

w = nf4.dequant_nf4(packed, absmax, codebook=your_codebook, dtype=torch.bfloat16)
w = w.view(shape)                 # the flattened weight
```

`codebook` is optional: the library's default is the standard table
(bitsandbytes' `create_normal_map()` with its own defaults, computed with
`torch.special.ndtri` in float64), which is what `NF4WeightStore`'s
`_nf4_codebook` computes too -- but passing the trainer's own table makes the
comparison exact by construction, and it is what the gate should be pointed
at. The double quantization stays on the trainer's side for now; it is a
small tensor per layer and the kernel does not do it yet.

## What it would be worth

| | int8 base (built) | NF4 + a fused dequant |
|---|---|---|
| storage for 2.17 B params | 2.17 GB | 1.26 GB |
| forward | int8 GEMM: parity with a fair bf16 linear, 15 % faster at M=1024 | dequant (14 ms) + bf16 GEMM: bf16 speed, i.e. **slower than the int8 path** |
| backward `dL/dx` | weight-only dequant (exact) or the int8 variant (1.33x, 1.35 % error) | dequant (14 ms) + bf16 GEMM: bf16 speed, and the gradient carries the **nf4 weight error (~1-2 %)**, not the int8 one (0.9 %) |
| extra VRAM in the step | none beyond the storage | none beyond the storage, *if* the dequant is fused (a torch chain would add ~16 bytes/param of scratch) |
| captures in a device graph | yes (verified) | yes, if the dequant is a single kernel with no temporaries |

So NF4 is worth it **only when the memory is the binding constraint and the
speed is not** -- and then it is worth about 0.9 GB against the int8 base
that already exists, at the price of a slower forward and a noisier
gradient. That is a real trade, and it should be made deliberately, per
model, not by default.

The hybrid is not worth it: storing nf4 and re-quantizing to int8 per step
would add a dequant *and* a quantize pass (~5 bytes/param) and would carry
the nf4 error anyway.

## What the library would need to add, and what the trainer would do with it

Library side (this repository, `xpu_kernels.triton` or a new
`xpu_kernels.nf4`):

* a fused `dequant_nf4(packed, absmax, dtype)` -- bit-exact against the
  trainer's current chain, behind the usual gate and env switch;
* optionally a packer (`quantize_nf4`) for the one-time conversion, which is
  a read of the bf16 weight and a write of the packed data: ~30 ms for 2.17 B
  parameters, once;
* the same capture-safety rules the INT8 path now has: the dequant is a
  single kernel with no temporaries, so it is safe inside a capture, and the
  gate must be latched before the capture starts.

Trainer side (not touched here):

* the `NF4WeightStore` would keep the *packed* bytes and the quantized
  absmax, and hand them to the fused kernel instead of running the chain;
* the per-layer peak becomes the bf16 output only;
* the decision of *which* layers are nf4 (the quantized subset is 2.17 B of
  the UNet's 2.57 B) and the fallback for the rest (the 3x3 convs and the
  narrow linears stay bf16) stays where it is.

## A second number that decides a feature: the weight dequant

The trainer's `Int8LinearFn` rebuilds the dense weight for every layer of every
backward (`frozen_weight` rebuilds it; only merge and DoRA need it, both
refuse). The cost of that rebuild was the open question in
`rework/72-kernels.md`: *"one dequant of the quantized layers per backward
(their figure: 2.6 ms per 25 M parameters, so roughly 0.2 s for the 2.17 B
here if it ran at that rate: to be measured, it is the number that decides the
feature)"*.

**Measured: 215 ms per backward for the 2.17 B quantized parameters.** That
was the torch chain, and it is the number that decided against the feature.

It is now **63 ms**: the library has a fused weight dequantizer
(`xpu_kernels.int8.dequant_weight`, a new SYCL kernel in `libxpuint8.so`),
bit-identical to the chain and 3.4x faster, because it moves 3 bytes per
parameter instead of fifteen. The exact (weight-only) backward is affordable
now; the int8 backward (`mode="int8"`) needs no dequant at all.

## One more thing worth knowing before the port

The nf4 error is larger than the int8 error, and the trainer's own probe
compares the *base* against itself. A base that is nf4 is a different
function from a bf16 base by more than an int8 base is: ~1-2 % of the weight
against 0.9 %. The loss curve of an nf4-base run is not comparable with a
bf16-base run, and the same is true of the int8 base but with a smaller
number. Whichever storage is chosen, the baseline has to be re-taken with
that storage, not carried over from the bf16 run.
