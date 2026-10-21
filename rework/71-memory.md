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
| Memory is claimed by name and purpose (`claim`); the ledger lists every claim, what no claim covers, the allocator's free memory (returnable segments vs pieces stuck between live blocks), this process's memory outside the allocator and other programs; each row is one slice of `device_used` | none | `ResidencyManager.claim/ledger` |
| Long-lived tensors are placed first (weights, adapter factors, optimizer state, gradients held and zeroed in place) and the cache is handed back once (`settle`), so no step allocates between them and the allocator can return whole segments | none | `Optimizer.hold_grads`, `ResidencyManager.settle` |
| A probe (one and a few samples of the largest shape, nothing applied) gives two points of (latent pixels, megabytes); the planner fits a fixed part plus a part that grows with the pixels (a single point counted the fixed part per pixel and overstated big batches by up to 40%), lifts the line over every point seen, and each step's peak keeps it current; a batch predicted not to fit runs as the fewest equal chunks whose gradients add up to the batch's | a split batch runs in more, smaller passes | `StepPlanner`, `Trainer._probe`, `Trainer._window` |
| The starter's Bucketer pads every latent to a multiple of 32 px and the loss is masked there (old implementation's method): the model sees a few dozen shapes instead of hundreds, nothing is cut away, peak memory is that of the padded shape | some extra compute in the padding | Bucketer node, `Batcher` |
| Latent Cache encodes one image at a time with exact sizes (a `step` setting, default 8, can cut sides further), the controller is started by the first node that uses the device, and images are decoded on a background thread: every distinct shape and batch size compiles kernels on one CPU thread (seconds each, GPU idle), and encoder activations of a big batch fill the card | none | Latent Cache node, `LatentCache.build` |
| One margin (`memory.margin_mb`, 512): allocator cap = limit - what the driver holds outside the allocator (measured, re-measured after steps) - margin; the ceiling for evictions = limit - margin; the planner's reserve is only what the update holds at the peak. Before this there were four stacked margins (10% of the limit, 1 GB then 0.5 GB headroom, a 768 MB reserve, 15% on growth) that left a 10240 limit with about 7.6 GB. At the cap the allocator gives back its free blocks before failing; `make_room` also checks the cap, so a load that needs more than the cap leaves is refused with the ledger instead of failing in the allocator | none | `ResidencyManager._cap`, `make_room`, `Accelerator.cap_allocator` |
| A step that runs out of memory is run again once on the same batches after cache and garbage are released, while no update has been applied; the cap then leaves more room for the driver | one repeated step | `Trainer._step`, `ResidencyManager.recover` |
| Trainer option `tune_checkpointing` (off by default): before step 1 the probe is run at the starting share of checkpointed transformer blocks and at half of it, the cost per pixel is extended linearly, and the share goes down in quarters as far as the largest batch of the run still fits the allocator cap minus a reserve; each share picked is measured before it is kept. Fewer recomputed blocks means fewer launches (the step is launch-bound), at the price of activations | up to a third of the recompute, as far as memory allows | `CheckpointDensity`, `Trainer._probe` |
| Trainer option `checkpoint_by=measured` (with `tune_checkpointing`): the probe's backward runs the checkpointer's observer, which times each block's recompute and takes the allocated-memory delta as its activation size; the blocks are ranked by milliseconds per megabyte and kept un-checkpointed while the largest batch still fits (probe sizes scaled by the pixel ratio, margin included); the result is measured again and, if it does not fit, the lowest-ranked quarter goes back to checkpointing | the best recompute per megabyte instead of an even share; residual blocks can be kept | `BlockProfiler`, `BlockPlacement` |
| LoRA's small factor takes the scale, not the full-size branch output | one full-size multiply per layer | `adapters/layers.py` |
| Pressure levers between steps: cache every step, then park residents (offload after the step, reload before the next) | one transfer per parked resident per step | `ResidencyManager` |

## Open (port or build when measurement says it pays)

NF4 frozen weights (old `nf4_weight_store`, `nf4_lora_layer`; measured on the B580 as 16-20% slower for 180-1361 MB, so only where memory is the binding constraint); block swap; activation offload to the host; oneDNN
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

## What is kept between runs, and what is not

| Kept | Where | Condition |
|---|---|---|
| Latents of the images | dataset folder, keyed by VAE and size policy | always |
| Text encodings of *stable* prompts (`Lifetime.STABLE`: captions of the set, dropout results) | dataset folder, `conditioning/`, one file per encoder identity | optional (`persist`), bounded by `max_disk_mb`; stops writing when full |

Prompts carry their lifetime (`data.prompts.Prompt` is a `str` with a `lifetime`): a stage that draws from a large space
(`TagShuffle`, `TagSwap`, `CaptionPool`) makes the prompts it changes FLEETING, and those never reach the disk, though
they live in memory for the epoch. A route that generates prompts of its own (a classifier, a blend) hands out
`Prompt(text, Lifetime.FLEETING)`. Not kept: noise and timesteps (they are the randomness), activations (every update
changes them), kernel compilations (oneDNN does not persist them; the SYCL disk cache measured 2%), converted model files
(disk to disk).

## Reading the ledger

Every row of the ledger is a separate slice of what the driver reports, so the rows add up to `device_used`.
"Ours" and "other programs" are separated with the kernel's per-client statistics (`/proc/self/fdinfo`, the DRM
`drm-resident-vram*` keys); where the kernel does not report them the single row "outside allocator" says so. The
events that explain a change in the card's memory: `allocator_cap` (names each term of the cap), `cache_returned` (what
an `empty_cache` returned, and whether the card's free memory rose by as much or other users took it), `cost` and
`breakdown` (outside memory split, segment count, allocator cache flushes). `--memory-log` adds `process_mb`,
`active_mb`, `split_free_mb`, `segments` and `alloc_retries` columns.
