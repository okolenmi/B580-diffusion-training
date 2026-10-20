# Trainer manual: features by what they change

A lookup for the moment you want *more of one thing* (less VRAM, faster steps, better quality, more control over
data). Each entry gives the switch, what it buys, what it costs, and its status.

**Status**: `verified` = run on the B580 and behaved as described. `untested` = written and unit-tested in the
sandbox only; the matching entry in `TESTING.md` says how to check it. `not built` = declared or planned, does
nothing yet.

## How to set things

| Way | Example | Reaches |
|---|---|---|
| CLI option of `python -m next.run sdxl-lora` | `--batch 2 --max-shapes 16` | the common ones, listed below |
| `--param NODE.PORT=VALUE` (repeatable) | `--param trainer.tune_checkpointing=true` | any port of any node in the starter graph |
| `--set KEY=VALUE` (repeatable) | `--set memory.vram_max_mb=10240` | graph settings (budget, precision, runtime) |
| `--dump FILE` | writes the graph instead of running it | inspect every node name and port |

Starter node names (the left side of `--param`): `model images latents captions tag_swap buckets noiser batcher
conditioner adapter optimizer schedule weighting trainer trigger saver` (+ `balance` when `--time-balance` is on).
`--memory-log FILE.csv` samples memory four times a second; `--verbose` prints every event.

## Which lever for which problem

| I want | Look at |
|---|---|
| lower peak VRAM, same quality | `batch_area`, `max_shapes`, optimizer `fused`/`group_size`/`state_precision`, `--rank` |
| lower VRAM, accepting slower steps | `checkpointing` on, `attention_fraction` 1.0, batch 1 + `accumulation` |
| faster steps | `tune_checkpointing` (`checkpoint_by=measured`), fewer shapes (`max_shapes`), `batch_size` up while peak allows, `--persist-conditioning` |
| faster start | `--persist-conditioning`, latent cache reuse, `max_shapes` |
| fix weak spots in some noise range | `--time-balance`, `--balance-sampling`, noiser `mode`, `probe_every` to see them |
| less overfitting to captions | caption nodes (shuffle, swap, pool, dropout) |
| understand where memory goes | `--memory-log`, ledger tables, `outside`/`ours`/`other` rows |

---

## 1. VRAM (peak device memory)

Reading the numbers first. `device_used` is what the driver says for the whole card. Our share is
`process` (kernel per-client counter, Linux/xe). Other programs are `device_used - process` (measured here:
about 760-800 MB, flat). Inside our share, `allocated` is live tensors, `reserved` the allocator's cache (part of it
returnable), and the rest ("outside allocator") is context, compiled kernels and primitive scratch: 442 MB at
start, about 1.5 GB after the first forward/backward, creeping by ~4 MB per new shape.

| Lever | Switch | Effect | Cost / caveat | Status |
|---|---|---|---|---|
| Pixel budget per batch | `--batch-area X` / `buckets.batch_area` | Samples up to X times the base square's area run at full batch size; a batch holds at most `batch_size * X` base-areas of latent pixels in total (rounded down, at least 1 sample). Bounds the peak by pixels, not by shape luck. Example, batch 4, X 1.5, base 96x96 latents: a 96x144 sample (1.5x) runs batches of 4, 96x148 (1.54x) runs 3 (4*1.5/1.54), 96x192 (2x) runs 3 as well, 96x288 (3x) runs 2. | Smaller batches for big shapes = more steps per epoch for them; optimizer sees different effective batch sizes. 0 = off. | verified (batch 2, stable peak) |
| Fewer short batches | `--pack P` / `buckets.pack` (default 0.3; 0 = off) | Each shape's left-over samples (fewer than a full batch) are first moved into a larger existing shape's short batch, then combined into one batch in the smallest existing shape that holds them all, as long as no sample is more than P padding. Packing never creates a shape, so `max_shapes` still bounds how many the model meets. Short batches are slow the first time (their own kernels) and, with big batch sizes, many; this turns them into full ones. The batch-area limit still sets each shape's full size. Padding is masked out of the loss (see padding fill). | Padded samples cost the host shape's compute; attention and convolutions see the padding, so keep P modest. The remaining short batches are still a batch size of their own (a new compile each time they first appear); the `data:` line counts them. A batch is never larger than an existing one, so the peak does not rise. Different samples are packed every epoch. | untested |
| Few shapes | `--max-shapes N` / `buckets.max_shapes` (default 16) | Every new shape costs ~4 MB driver memory and several seconds of kernel compile the first time. Cheapest shapes are merged to a common padded size until at most N remain. Limit for 768 base, x2 max aspect, 4-latent padding multiple: 49. | Padding wastes compute in merged shapes (masked out of the loss). `padding_stats` and the `data:` line show the cost. 0 = no limit. | verified (8 shapes: ~7% mean padding; 49 or 0: ~2%) |
| Gradient checkpointing | `adapter.checkpointing` (CLI `--no-checkpointing` turns off) | Recomputes activations in backward instead of storing them: the biggest single saving. | ~+7.5k kernel launches per step; a launch-bound step gets slower. | verified |
| Share of blocks checkpointed | `adapter.attention_fraction` 0..1 | Fewer checkpointed transformer blocks = more activations kept = faster, more VRAM. Residual blocks always checkpointed. | Evenly spread blocks; fraction is a manual guess, see auto-tuner below. | untested |
| Auto checkpoint share | `trainer.tune_checkpointing=true` | Before step 1, measures the largest batch at several shares and picks the fewest-recompute share that fits the budget. Event `checkpoint_share`. | A few extra measuring passes at start. Use with an explicit `memory.vram_max_mb` for a margin. | verified at batch 1 (share 0.75, 7% faster); tight-budget crash fixed, retest |
| VRAM budget | `--set memory.vram_max_mb=N` | Allocator cap = limit - what other programs and our non-allocator memory hold - margin. The planner splits batches that would not fit; one OOM retry. | Too tight = more splitting/slowness. `auto` = whole card. | untested (cap and planner) |
| Return allocator cache | automatic (`cache_returned` event); `--set runtime.empty_cache_every_n_steps=N` (default 1) | Gives free cached blocks back to the driver so the cache's creep (7.4 -> 7.9 GB seen) does not look like growth. | Re-allocation costs a little time; 0 never empties. | untested |
| Fused optimizer | `--fused` / `optimizer.fused` | Updates each parameter when its gradient is ready: gradients never pile up. | No gradient clipping when fused. | untested |
| Optimizer grouping | `optimizer.group_size` (0 = all same-shaped, 1 = one at a time) | 1 uses the least temporary memory. | Slower (more small launches). | untested |
| Optimizer state | `optimizer.state_precision=int8_blockwise` | State at about a quarter of the memory. | Small precision loss. Adafactor (`beta1` 0) keeps least state; CAME in between. | untested |
| Adapter size | `--rank` / `adapter.rank` | Parameters, gradients and optimizer state scale with it. | Capacity. | verified |
| Fewer trained modules | `adapter.groups`, `include`, `exclude` | Fewer adapter parameters and less backward work. | Capacity/coverage. | untested |
| Encoders off the card | `conditioner.release_encoders` (default true) | Text encoders live on the host and visit the device only to encode. | Slower first epoch unless the conditioning store is on. | untested |
| Precision | `--set precision.denoiser/encoders/vae=auto|bf16|fp16|fp32`, `precision.autocast` | Lower precision = less memory. | fp32 is a big jump; quality risk with fp16. | untested |
| Accumulation instead of batch | `--accumulation N` | Same effective batch with the memory of one micro-batch. | More forward/backward passes; no larger batches' kernel efficiency. | verified |
| Allocator mode | env `PYTORCH_ALLOC_CONF=expandable_segments:True` | Could reduce stuck fragmentation. | Not known to work on XPU. | untested |

Not built (old tree had them, see `rework/71-memory.md`): NF4 frozen weights (measured 16-20% slower for
180-1361 MB), block swap to host, activation offload.

## 2. Speed

A step on this card is launch-bound (old measurement: ~22,300 kernel launches, ~48 us each; recompute adds ~7.5k).
So: fewer launches, fewer new shapes, fewer syncs.

| Lever | Switch | Effect | Cost | Status |
|---|---|---|---|---|
| Measured checkpoint placement | `trainer.tune_checkpointing=true` with `trainer.checkpoint_by=measured` | The probe's backward times each block's recompute and counts the memory of its activations; the blocks that save the most time per megabyte stop recomputing, residual blocks included, as far as the largest batch fits. Event `checkpoint_plan` lists the kept blocks with their MB and ms. | Same measuring pass as the share tuner, plus a timing synchronization per block in it. Keeps everything checkpointed when the profile shows no memory figures. | untested |
| Less recompute | `tune_checkpointing`, `attention_fraction` | Each skipped block's recompute is not run. | VRAM. | untested |
| Few shapes | `max_shapes` | First sight of a shape ~3.85 s; fewer shapes = fewer slow steps (the step line marks `new`). | Padding. | verified |
| Bigger batch | `--batch` | Launches are shared by more samples. | VRAM; use with `batch_area`. | verified |
| Conditioning store | `--persist-conditioning` / `conditioner.persist` | Second run neither loads encoders nor encodes. `conditioner.max_disk_mb` (4096) caps writes. | Disk. | untested |
| Latent cache | automatic, next to the dataset | Encoded once per (`base`, `mode`, `max_aspect`, `step`). Cold build was ~108 s. | Disk. | verified |
| Latent encode batch | `latents.batch_size` (default 1) | Each distinct batch size compiles its own kernels; 1 is usually fastest overall. | - | verified (old tree) |
| Prefetch | `batcher.prefetch` (default 2) | Batches built ahead of the step. | Host memory. | verified |
| Fused/grouped optimizer | `fused`, `foreach`, `group_size` 0 | Fewer launches in the update (0 = all same-shaped at once). | `fused` no clipping. | untested |
| Graph replay | `--graph` / `trainer.graph=true`, `trainer.graph_shapes` (8) | After a shape has been seen 3 times (eagerly) its forward + loss + backward is captured as a device graph and replayed: about 2x faster steps in the old measurements (launch-bound step), +37% steps/s end to end. Graphs share one memory pool, so the cost is the largest graph (old tree: ~0.5-1.5 GB), not the sum. Uses the memory-efficient attention kernel (flash cannot be captured). | Needs: no dropout, no timestep gate, non-fused optimizer (all refused loudly); chunked batches, the probe, and shapes past `graph_shapes` run eagerly. Count shapes with `max_shapes`: each graph is a capture (seconds). Competes with `tune_checkpointing` for the same spare VRAM. | untested |
| Timing | `trainer.timing=true` | Per-phase times; synchronizes, so it slows a bit. Turn off for speed. | - | untested |

## 3. Quality and training behaviour

| Lever | Switch | Effect | Status |
|---|---|---|---|
| Learning rate | `--lr`, schedule nodes (`ConstantLR`, `CosineLR` with `min_fraction`, `WarmupLR` `steps`/`start`) | Standard. | verified |
| Train on top of a LoRA | `Merge Adapter` node between Load Model's denoiser and Inject Adapter (chain several; `strength` scales, negative subtracts) | The saved kohya LoRA is added into the frozen weights when the denoiser loads (`W += strength*alpha/rank * up@down`, rounded once), so the new adapter trains on the merged model. Refuses a file that matches nothing. | A little time at load. The merge is not saved; export the new adapter and merge both when using it. | untested on a real LoRA (equivalence verified on a tiny UNet) |
| Conditioning blend | `--param blend.strength_max=0.5` (and `strength_min`, `probability`, `partners`, `segments`) on the starter's `Blend Conditioning` node | Each sample's text conditioning becomes `caption*(1-s) + partner*s`, s drawn per sample and step from min..max, with chance `probability`; the partner defaults to the empty prompt, or one of the lines of `partners`. Gives the cross-attention a continuum between a caption and nothing instead of one point per caption (or jumps with dropout). Random draws are made after the cached encode, so nothing random is cached. Strength 0 is bit-exact. | Nothing measurable; partners are encoded once up front. Not shown to fix anything by itself. | untested |
| Optimizer choice | `--optimizer AdamWNode|AdafactorNode|CAMENode` | AdamW is the baseline; Adafactor `clip_threshold`, `scale_parameter`; CAME `beta3`. | verified (AdamW) |
| Gradient clip | `optimizer.grad_clip` | Largest total norm; 0 off. Not with `fused`. | untested |
| Adapter kind | `adapter.kind=lora|dora`, `scaling=classic|rank_stabilized`, `alpha`, `dropout` | `rank_stabilized` uses alpha/sqrt(rank) so higher ranks do not shrink. | untested (dora) |
| Region/range control | `adapter.regions`, `adapter.gate*` | Scale or switch off parts of the net; fade the adapter outside a timestep range. | untested |
| Timestep distribution | `noiser.mode=uniform|low|mid|high|logit`, `low`, `high`, `values` | Where noise levels are drawn. | verified (uniform) |
| Parameterization | `noiser.parameterization=eps|v`, `zero_terminal_snr` | Prediction target. | untested |
| Loss weighting | `weighting` node | Per-timestep loss weights. | verified |
| Time balance | `--time-balance off|normalize|speed|dro` (node `balance`) | Reweights the loss per third of the timestep range (weights clipped to 0.25-4x). Each third is compared with its own loss over the first 10 reports: `normalize` weights by 1/that start loss (evens out loss scales), `speed` raises the weight of a third that descends slower than the others, `dro` gives the most weight to the third that has improved least relative to its start. Step line prints the weights; `normalize` can pin the smallest-loss third at the clip, it is a strong setting. | normalize/dro verified; speed got a `leak` (default 0.05, `balance.leak`) after it ratcheted to the clip |
| Balanced sampling | `--balance-sampling P` | Draws timesteps more often from thirds whose loss is high relative to their own start (chance ~ (loss/start)^P; 1 is a start). | works with dro; step line now prints `draws a/b/c` |
| Probe | `trainer.probe_every=N`, `probe_items`, `probe_points` | Every N steps compares adapter vs frozen model on fixed samples per third: `probe_rel_t_*` above 1 = the adapter makes that range worse. Diagnostic only; costs a few extra forwards. | untested |
| Padding fill | `--pad-fill zero|mirror` / `buckets.fill` (default zero) | What the padded part of the network input holds. `zero`: a blank strip, noised like the rest. `mirror`: a reflection of the sample's own *noisy* input (same noise values, so no second view of the picture and no extra information); normalisation and attention then see image-like content and the seam is continuous. The loss ignores padding in both. Which trains better is not known: compare two LoRAs. The time probe still uses zero padding. | untested |
| Padding and masks | `buckets.multiple` | Pad latents up to a multiple, loss masked on pad; no pixels cut. | verified |
| Shapes | `latents.mode=fit|center_crop|pad|stretch`, `max_aspect`, `step` | How images become latents; `fit` keeps aspect, cutting at most `step-1` px. | verified |
| Captions | nodes `Captions` (`channel` text/tags/desc), `TagShuffle` (`keep_first`, `drop`), `TagSwap` (`random`/`walk`), `CaptionPool` (`n_min`, `n_max`, `dividers`), `CaptionDropout` (`p`, `substitute`/`empty`) | Prompt variety per epoch; all seeded by `captions.seed`, so epochs repeat identically per seed. | verified (basic) |
| Seeds | `--set seed.base=N`, or `batcher.seed`, `captions.seed` | Data order, padding offsets, timesteps, noise. A node seed left empty derives from the base and the node id. | untested (base) |
| Run length | `--steps N` and/or `--epochs N` (`trainer.steps`, `trainer.epochs`) | An epoch is one pass over the batcher's batches (with accumulation, one step is `accumulation` batches). With both set the run ends at whichever limit comes first; 0 turns a limit off (`--epochs` alone sets steps to 0). A cosine schedule without `total_steps` follows the run's length. | verified (CPU) |
| Save per epoch | `--save-every-epoch N` (node `EveryEpochNode`) | Saves at the last step of every Nth epoch, in addition to `--save-every` (Every N Steps); wire either or both to the saver. Step lines show `epoch N` (`end` on its last step). | verified (CPU) |
| Resume | `trainer.start_step`, `batcher.start_epoch`, `LoadAdapter` | Schedule, data order and prompts continue from there. | untested |

## 4. Data and shapes

| Item | What it does |
|---|---|
| Auto caption | `--auto-caption` (model and options: `--param autocaption.model_id=...`, `prefix`, `suffix`, `lowercase`, `limit`, `overwrite`, `channel`) | Before training, a Hugging Face image-to-text model (BLIP base by default) writes captions into the dataset's library for images whose caption is empty, so a captionless set stops giving cross-attention one constant conditioning. Resumable (only empty captions are written), commits as it goes, frees the model afterwards. Backends are pluggable (`register_captioner`). | Time for one pass over the images, once; downloads the model on first use. | untested on a real model (logic verified with a stub) |
| `latents.base` / `--resolution` | Short side in pixels (768 = SDXL-ish low budget, 1024 full). |
| `latents.max_aspect` (2.0) | Longest side is at most this times base. |
| `buckets.multiple` | Latent padding multiple (4 latents = 32 px). |
| `buckets.max_shapes` | See VRAM/Speed. |
| `buckets.batch_area` | See VRAM. |
| `buckets.pack` | See VRAM (fewer short batches). |
| `buckets.keep_incomplete` | Train on a short last batch (a shape of its own: costs a compile) or drop it. |
| `batcher.shuffle` | Shuffles batches. Same-shape batches arrive in clumps of up to 4. |
| `data:` line (console) | Epoch size, shapes, short batches, packed and dropped samples, padding share (of what the plan really runs); check it after changing any of the above. |

## 5. Diagnostics

| Tool | Gives you |
|---|---|
| Gradient alignment | `--param trainer.probe_every=50 --param trainer.probe_alignment=true` | At each probe, the gradient of the probe loss per third of the timestep range: `gc_norm/share` (which third dominates), `gc_cos_a_b` (negative = a step that helps one hurts the other), `gc_align_t_*` (cosine with the sum; <= 0: the combined update does not help that third), `gc_self_t_*` (split-half noise floor: read the other numbers relative to it). Explains `probe_rel_*`; not a trajectory. | One backward per probe point at every probe (autograd.grad, so the optimizer's gradients are untouched). Raise `probe_items`/`probe_points` to lift the noise floor. | untested on a real network (closed forms verified) |
| `--memory-log FILE.csv` | Columns: seconds, label, allocated, reserved, peak_allocated, device_used, process, active, split_free, segments, alloc_retries (MB). Labels are node/step/task names. |
| Ledger tables (console) | Additive slices of `device_used`: ours (allocated, cache, outside) and other programs. |
| Step line | `[BxHxW new]`, loss, weights, probe, timings. |
| Events | `allocator_cap`, `cache_returned`, `checkpoint_share`, `cost`, breakdown. With `--verbose`. |
| `split_free` / `segments` / `alloc_retries` | Fragmentation: free pieces stuck between live blocks; retries mean the allocator had to free and try again. |
| Task lines | Named long processes (latent build, conditioning) with progress. |

## 6. Persistence on disk

| What | Where | Control |
|---|---|---|
| Latent cache | next to the dataset | `latents.*` keys; rebuilt when they change |
| Conditioning store | next to the dataset | `persist`, `max_disk_mb` |
| Adapters | `--output`, every `--save-every` steps | `saver.name`, `format` (kohya, native) |

## 7. Declared but not implemented

These graph settings exist in the registry and appear in the editor, but nothing reads them yet:
`runtime.graph_capture` (use the trainer's `graph` port instead), `runtime.block_swap`, `memory.ram_max_mb`. Setting them changes nothing.
Read by code: `memory.vram_max_mb`, `memory.vram_min_mb`, `memory.strict`, `device.default`, `precision.*`,
`runtime.residency.safety_margin`, `runtime.residency.calibration_steps`, `runtime.empty_cache_every_n_steps`,
`runtime.profile` (turns the trainer's `timing` on), `conditioning.cache_mb` (size of the encoded-prompt cache,
auto = 4096 MB; the Conditioner's own `cache_mb` port wins), `conditioning.prewarm` (off = encode prompts when first
needed instead of up front), `seed.base` (nodes with an empty seed derive theirs from it and their id; -1 leaves
them at 0).

Planned ports (see `rework/PORTING.md`): NF4 and int8 frozen weights (to use the owner's rewritten kernels).
