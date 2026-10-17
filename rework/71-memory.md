# 71. Device memory: sources, levers, and what is still open

The monitor's data is `ResidencyManager.breakdown()` (reported as the `memory` stream, and printed by `python -m next.run`):
each resident (model, optimizer state, encoders), other live tensors, the step's activation peak, free blocks the
allocator keeps, and what the driver holds outside the allocator. `device_used` (from the driver) is the number that
predicts out-of-memory; the allocator's own peak understates it.

## Levers in place

| Lever | Cost to speed | Where |
|---|---|---|
| Text encoders on the host except while encoding (`OnDemandKit`); the device cache is emptied when they leave | one round trip per epoch | `engine/runtime/encoders.py`, Conditioner node |
| VAE released and the device cache emptied after the latent cache is built | none | Latent Cache node |
| Frozen weights in bf16, loaded straight to the device (meta construction, no fp32 copy) | none | `families/sdxl/denoiser.py` |
| Activation checkpointing: residual blocks always, transformer blocks by `attention_fraction` | recompute of the chosen blocks | `engine/runtime/checkpointing.py` |
| Fused optimizer (no gradient pile-up), grouped updates, int8 optimizer state | small | `engine/optim` |
| Allocator cache emptied after any offload, and whenever cached blocks push the card past the ceiling (the driver's own allocations for a new shape cannot make the allocator release them) | one synchronise, only when needed | `ResidencyManager` |
| Pressure levers between steps: cache every step, then park residents (offload after the step, reload before the next) | one transfer per parked resident per step | `ResidencyManager` |

## Open (port or build when measurement says it pays)

NF4 frozen weights (old `nf4_weight_store`, `nf4_lora_layer`); per-block checkpoint placement from measured block costs
(old `block_profiler`, `checkpoint_placement`); block swap; activation offload to the host; graph capture; oneDNN
primitive-cache sizing and thrash detection (old `xpu_env`, `cache_thrash`).

## Carried over from the old implementation (measured on the B580, `nodes/xpu_env.py`, `docs/known-issues/open.md`)

* A new latent shape costs about 3.85 s the first time it reaches the GPU (kernel and primitive creation). With
  oneDNN's default primitive cache (1024 entries) a shape that comes back also costs that much again; with 2048 it
  costs the normal step time. The requirement is about 29-35 primitives per distinct latent shape, the cache is a
  ceiling that costs no memory, so `size_cache_for_shapes` sizes it from the dataset. First sightings cannot be cached
  across runs; fewer distinct shapes (bucketing with a multiple above 1) is the only lever. A short test run on a
  mixed-aspect dataset is therefore dominated by first sightings and looks much slower than a long run.
* The SYCL variables in `engine/xpu_env.py` keep the kernel cache in memory and avoid stalls; the entry points
  (`next/run.py`, `scripts/xpu_smoke.py`) set them before anything touches the device.
* "1024" means the **short side** is 1024 px. The long side keeps the aspect ratio, up to `max_aspect` times the base
  (2048 by default) and longer images are split into several crops. A 2:3 photo is therefore 1024x1536, about 1.5 times
  the pixels of a square 1024, and its activations are larger by the same factor. Lower the base to lower memory.
