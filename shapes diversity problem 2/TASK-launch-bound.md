# TASK: the training step is CPU-launch-bound (follow-ups to the shape-stall work)

Repository `okolenmi/B580-diffusion-training`, `main` at `61c172b` or later. Read
`shapes diversity problem/MEASURED-shape-stall.md` first (the "single-threaded stall"
section) and run `count_launches.py` (copy it to `scripts/`). Same working rules as the
earlier task files: one commit per item, measurements in the commit body, never weaken a
test, say plainly what was only tested on CPU.

## Established (what the numbers say)
* Measured by the project: 1.00 CPU cores busy in fast steps too, CPU time per step ==
  wall time per step, GPU 25-30%. The whole run is bound by one thread issuing launches.
* Measured here with `count_launches.py` (meta device, the project's own UNet and LoRA
  layers, batch 2, latent 64x64): **22,286 kernel launches per production step**:
  base model 6,726 (30%), LoRA branch +8,095 (36%), checkpoint recompute +7,465 (33%).
  `aten::_to_copy` (casts) alone is 5,338. At the measured 1.06 s CPU/step that is
  ~48 us per launch, normal for an eager PyTorch XPU launch, so the count explains the
  time. Step time is therefore (launches) x (~48 us); shape work only removed the extra.

## L1 Fix the conditioning of bucketed (padded) samples  [quality bug]
`nodes/train/managed.py:505` and `step_pipeline.py:265` pass
`height=x_t.shape[2]*8, width=x_t.shape[3]*8` to `encode(...)`. For a bucketed batch that is
the PADDED size, so the model is told "768x512" about content that may be 520x480, and one
value is used for the whole batch even though its samples have different true sizes.
**Do:** derive each sample's true size from its `valid_mask` (extent of the valid region) and
pass per-sample size embeddings (`resolution_embedding` per distinct true size, concatenated
into `y`); unbucketed batches must produce byte-identical conditioning to today.
**Tests:** padded sample -> conditioning equals that of the same sample unpadded; mixed true
sizes in one bucket -> per-sample rows differ; unbucketed path unchanged (compare tensors).

## L2 Measure whether bucketing hurts quality, and tune the multiple
Pad fraction at multiple-of-32 is large for some samples (up to ~40-50% of the canvas).
**Do:** (a) report the pad fraction per sample and its distribution when bucketing is enabled
(log once at build); (b) run `shape_bucket_multiple` 16, 24 and 32 on the real dataset and
record first-sighting seconds, steps/s and pad fraction for each (the planner in
`shape_policy.py` lists candidates); (c) compare loss on a fixed UNPADDED holdout after N
steps with and without bucketing, same seed. Keep the default off until (c) is reported.

## L3 Runtime detector for primitive-cache thrash
The sizing constant (35 primitives per shape) was measured for one configuration; rank,
DoRA, targets and optimizer change it. Step shape is already logged. **Do:** in
`MonitoringPhase`, track step time for repeats vs revisits of a shape; if the revisit median
exceeds 1.5x the repeat median after 3 revisits, log ONE warning naming
`ONEDNN_PRIMITIVE_CACHE_CAPACITY` and the observed ratio. **Test:** synthetic step times.

## L4 Heterogeneous captions in one batch  [the structural lever]
`manager/loader.py` forms batches per `(prompt, neg_prompt, size)` and the trainer encodes ONE
prompt per batch, so images with unique captions cannot share a batch. Because the step is
launch-bound, step time is almost independent of batch size: batch 1 -> 8 would give close to
8x images/s until the GPU becomes the limit. **First measure, no code:** run `hw_validate.py`
with `--batch` 1, 2, 4, 8 on a dataset whose images share captions; record step time and
peak memory. If step time stays flat, do the change: per-sample `ctx_emb`/`y` (this also
subsumes L1), loader groups by size bucket only (prompt no longer in the key), the text cache
keyed per prompt. Keep the old path behind a flag.

## L5 Cut launches, cheapest first (use `count_launches.py` as the acceptance number)
1. **Checkpoint recompute is 33% of launches.** `enable_attention_block_checkpointing(fraction=...)`
   already exists. For the small bucketed shapes try fraction 0.5 and 0 and record peak
   memory; choose the largest uncheckpointed fraction that fits the ledger budget, per shape.
2. **LoRA branch: 5.3k casts.** `LoRALinear.forward` casts the activation to fp32 and back,
   recomputes `lora_B.T * scaling` on every call, then adds. Try: cast the small A/B once per
   step instead of the activation twice per layer; fold `scaling` into the matmul
   (`torch.addmm(..., alpha=)`); fuse q/k/v (shared input) -- `GroupedLoRALinear` was an
   unfinished attempt and is never instantiated. Acceptance: fewer launches in
   `count_launches.py` AND loss parity over 200 steps on the same seed (bf16 vs fp32 LoRA
   compute is a numerics change: report it, do not assume it).
3. **XPU graph capture.** Bucketing leaves 3 static shapes. `torch.xpu.XPUGraph` exists in
   torch 2.14; **first check `hasattr(torch.xpu, "XPUGraph")` on the 2.12.1+xpu build.** If
   present, probe feasibility in isolation: capture one UNet forward+backward at one shape
   with static input buffers and time replay vs eager. Known hazards: dropout RNG, the
   custom checkpoint function's `fork_rng`, the memory pool. Report the replay speedup
   before building anything.

## Report
One line per item (`done | partial | skipped`), commit, the measured before/after, and
anything only verified on CPU.
