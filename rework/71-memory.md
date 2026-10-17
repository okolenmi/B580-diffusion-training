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
| Memory is claimed by name and purpose (`claim`); the ledger lists every claim, what no claim covers, the step's activation peak, the allocator's free blocks and what lives outside it | none | `ResidencyManager.claim/ledger` |
| Long-lived tensors are placed first (weights, adapter factors, optimizer state, gradients held and zeroed in place) and the cache is handed back once (`settle`), so no step allocates between them and the allocator can return whole segments | none | `Optimizer.hold_grads`, `ResidencyManager.settle` |
| A probe (one sample of the largest shape, nothing applied) measures the cost of a latent pixel before step 1; each step's peak keeps it current; a batch predicted not to fit runs as the fewest equal chunks whose gradients add up to the batch's | a split batch runs in more, smaller passes | `StepPlanner`, `Trainer._probe`, `Trainer._window` |
| The starter's Bucketer pads every latent to a multiple of 32 px and the loss is masked there (old implementation's method): the model sees a few dozen shapes instead of hundreds, nothing is cut away, peak memory is that of the padded shape | some extra compute in the padding | Bucketer node, `Batcher` |
| Latent Cache encodes one image at a time with exact sizes (a `step` setting, default 8, can cut sides further), the controller is started by the first node that uses the device, and images are decoded on a background thread: every distinct shape and batch size compiles kernels on one CPU thread (seconds each, GPU idle), and encoder activations of a big batch fill the card | none | Latent Cache node, `LatentCache.build` |
| Allocator capped at the limit minus what the driver holds outside it (measured, plus headroom for a new shape's kernels); at the cap the allocator gives back its free blocks before failing. The old `apply_allocator_backstop` did the same | none | `ResidencyManager._cap`, `Accelerator.cap_allocator` |
| A step that runs out of memory is run again once on the same batches after cache and garbage are released, while no update has been applied; the cap then leaves more room for the driver | one repeated step | `Trainer._step`, `ResidencyManager.recover` |
| LoRA's small factor takes the scale, not the full-size branch output | one full-size multiply per layer | `adapters/layers.py` |
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
