# The SDXL step, measured in this tree: what the 1.3 s is made of

*Measured 2026-10-21 on the B580, with the trainer's own command line
(`--resolution 512 --batch 4 --rank 16 --batch-area 1.5 --param
adapter.base_precision=int8 --param buckets.max_shapes=4`, the `my_set`
dataset).  Every run is the same command with one thing changed; the numbers
are the minimum over the steps of one shape (3×100×64 latents).*

## The reason the step is slow: the capture cannot use the flash attention

`next/engine/step/graphed.py` captures with
`sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)`, because "the flash kernel
cannot be captured".  On XPU that backend is **not supported** — torch says so
("SDPA Memory Efficient Attention backend is not supported on XPU, falling
back to math backend") — so the captured step runs the **math** attention
while the eager step runs the **flash** one.

The device cost of that substitution, measured on this trainer's own UNet
(512×512, batch 1, device time by events):

| attention backend | the forward | forward + backward |
|---|---|---|
| the default (**flash**) | 120.8 ms | 402.9 ms |
| `EFFICIENT_ATTENTION` (→ math on XPU) | 163.4 ms | **558.5 ms** |
| `MATH` | 164.3 ms | 548.3 ms |

**The math attention costs +36 % of the step's device time.**  The capture
saves the host time (a few tens of ms here) and pays 36 % of the device time
for it.  That is why the graph is *slower* than the eager path, and why
"without the graph it runs faster".

The flash kernel cannot be captured because it uses the work-group scratch
memory, and the SYCL Graph extension does not support that feature:

```
RuntimeError: The sycl_ext_oneapi_work_group_scratch_memory feature is not
yet available for use with the SYCL Graph extension.
```

That is a driver/runtime limitation, not a trainer's bug — and it is worth
reporting upstream (the same way the `reduce_over_group` crash was).

## The four cells

| base | eager (no graph) | replayed (graph) | the graph's effect | live memory |
|---|---|---|---|---|
| **bf16** | **1134 ms** | 1324 ms | **−14 % — the graph is slower** | 5546 MB |
| **int8** | 1550 ms | **1452 ms** | +6 % | 3489 MB |

And the int8/bf16 ratio in each mode: **+37 % eager** (the int8 path launches
about twice the kernels, so the eager step is host-bound) and **+10 %
replayed** — the same +10-12 % the kernels' own probe measures on this UNet.
The int8 base is not the problem; it is a 2.06 GB trade.

## What to do

* **Do not force `EFFICIENT_ATTENTION` on XPU.**  It is not supported, it
  falls back to math, and math is 36 % of the step.  Either let the capture
  use whatever the eager path uses (and let the capture fail where it cannot),
  or compare the replay against the eager step and drop the graph when the
  replay is slower — the machinery for "a shape that cannot be captured trains
  eagerly instead" is already there; it just does not know about this.
* Until the flash kernel can be captured, **the graph is a pessimization on
  this card**: the eager path (flash) is the fastest thing here, 1134 ms
  against 1324 ms.
* `buckets.max_shapes` is still the difference between a graph and no graph:
  without it every step's shape is new, the warm-up never completes, each new
  shape costs 4-5 s, and the graph is captured and **never replayed**.
* The next lever is the device side: the GEMMs are 45 % of the step's device
  time, the convolutions 22 % (334.8 M of the UNet's parameters, and ConvRot
  cannot reach a conv), the attention 5 % (but 13× that with the math
  fallback).
