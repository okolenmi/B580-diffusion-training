*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 9 -- the Resources Controller route's own trainer, built around residency instead of gated onto the main route's

**Goal:** unblock the one thing Phase 6 left open (`LoRATrainingConfigNode`'s
`trainer` output had nowhere to plug in), and build the Resources
Controller route as a genuine second, memory-optimized alternative to
the existing main route -- with its own training step loop, not the
main route's loop reused behind a stricter gate. See "First attempt,
reverted" below for why that distinction is the whole point of this
phase, not a style preference.

### First attempt, reverted

- sharing one loop meant the "new route" was the main route's loop
  with a stricter gate in front of it, not an alternative memory
  design.
- Phase 9's own trainer node no longer has those ports to adapt
  into -- it takes `trainer` directly, so there's nothing left for
  that node to bridge.
- The original design released text_encoder/optimizer
  unconditionally, every step, regardless of whether the stated
  budget ever needed it.
- Ongoing escalation, not calibrate-once.

### The actual design: `nodes/train/managed.py`

- `BackwardAndOptimizerStepPhase`: loads `optimizer`, runs backward
  (and `.step()` for a non-fused optimizer), releases it -- one phase,
  one residency window spanning both, not two. Checked directly against
  `ComposedFusedOptimizerHandle` (`nodes/optimizer/composed_fused.py`):
  a fused optimizer's real update happens inside a backward-pass hook
  (`_on_grad_ready()`, fired per-parameter as each gradient becomes
  ready *during* `backward()` itself), not in a separate call after --
  `FusedOptimizerHandle.step()` is a no-op. So optimizer state has to
  already be resident before `backward()` starts, for a fused optimizer
  -- not just before some later step()-shaped phase. A non-fused
  optimizer doesn't strictly need state resident during backward, only
  during its own `step()` call after, but loading it slightly earlier
  than strictly required is a small, deliberate over-inclusion, traded
  for one rule correct for both cases rather than a fused/non-fused
  branch in the residency logic itself.

A real run reported ~0.36 steps/sec against this route vs. ~1.7 steps/sec
on the main route (same settings, AdamW) -- a ~4.7x slowdown -- with
peak reserved VRAM around 9.0GB against a stated 12500MB budget the
whole time.

## Third addendum: activation memory is the ceiling residency management and weight precision can't touch

The report that surfaced the above also included the actual numbers
behind an OOM: 10218MB reserved, of which the three residents this
route tracks accounted for only ~4806MB combined (model 3061MB +
optimizer 184MB + text_encoder 1561MB) -- more than half the real usage
was activation memory (forward/backward intermediate tensors), which
neither `AdaptiveResidencyController` nor NF4/Int8 weight quantization
touches at all. Releasing every candidate this controller manages
(1745MB combined, in that report) was never going to be enough headroom
against a multi-GB activation spike on its own.

The same report asked why switching from NF4 (frozen base) + Int8
(optimizer state) to bf16 + fp32 only changed measured usage by ~1GB,
much less than the ~4x difference their own storage footprints would
suggest. Real, and not a bug: both `NF4WeightStore` (`nodes/model/
nf4_weight_store.py`) and `Int8BlockStateStore` (`nodes/optimizer/
state_store.py`) dequantize to a real, transient full-precision buffer
on every use -- `footprint_bytes()` deliberately doesn't count that
buffer (both classes' own docstrings: it's the caller's, freed right
after, not storage either class holds) -- so the *resting* footprint is
genuinely ~4x smaller, but the *peak-during-compute* footprint (what
actually determines whether a step OOMs) is much closer to the
unquantized case, because the transient buffer still has to exist for
that one use. This is inherent to this whole class of technique
(bitsandbytes/QLoRA has the same characteristic), not specific to this
project's own implementation. (Also, for the record: `unet_weight_store`
only has two real choices, `"bf16"` and `"nf4"` --
`nodes/model/lora_training_config.py`'s own `_UNET_WEIGHT_STORE_CHOICES`
-- there is no `"nvfp4"` option in this codebase; a value outside those
two would have raised immediately via `Port.choices` validation, so
whatever was actually selected and produced these numbers was `"nf4"`.)

## Fourth addendum: `prewarm_text_encoder` -- the text encoder leaves VRAM for good

The reports above counted the text encoder (1561MB) as a permanent
resident this route carries for the whole run, and the calibration design
then concluded, correctly given what existed at the time, that nothing
could be done about it cheaply: releasing it meant
re-uploading it whenever the next encode needed it, and the ordering
rule (candidates released smallest-footprint-first) meant any pressure
sufficient to reach the encoder had already dragged the always-needed
optimizer out with it -- measured at -54% throughput in the 2026-09-28
floor-lever sweep (`docs/known-issues/open.md`, `--budget 8000`).

- **Build order matters twice.** Discovery + wrap/bind happen before
  `resource_control.register("text_encoder", ...)` (discovery reads
  only `batches`, no encoder, no handle); the warm pass itself happens
  *after* registration, because warming an empty cache takes misses and
  each miss calls `ensure_loaded("text_encoder")`, which needs the name
  registered first. Fresh wrap sizes `max_entries` to the discovered
  key count so the warm pass can't evict itself; an existing
  `cache_text_encoder` wrap is kept and gets its handle late-bound via
  `CachingTextEncoder.bind_resource_control()` (LoRATrainingConfigNode
  runs before `resource_control` is anywhere in scope, so that wrap
  can't have had one).
- **Measured** (managed route, batch 2, dataset 1024, 40 steps,
  budget 11500): floor 7888 -> 6327 MB allocated at step 0, peak
  reserved 9268 -> 7666 MB, throughput 0.716 -> 0.789 steps/sec
  (+10%: the per-step CLIP forward is gone too), loss curve healthy.
  The controller's calibration now measures a peak that already
  excludes the encoder, and a 0-footprint candidate is one it stops
  considering -- no ordering change to `AdaptiveResidencyController`
  was needed after all.
- **It also unblocked the fraction sweep, and settled it.** With the
  floor cut applied *before* calibration (the budget path could only
  release after 3 fully-resident calibration steps, so it never got the
  chance -- `M_frac75low` OOM'd during calibration), density 0.75
  completes: 40/40 steps, peak 10922 MB, ~300 MB under the wall. It is
  not a speed win, though -- 0.759 vs 0.789 steps/sec at density 1.0:
  skipping 25% of attention-block recompute came out slightly slower
  while adding 3.3GB of activation residency. Recomputing these blocks
  is cheaper than carrying their activations. Density 1.0 (full
  checkpointing) + prewarm is the configuration that wins on both axes,
  and the full numbers live in `docs/known-issues/open.md`.

## Fifth addendum: the loop itself -- accumulation, warmup, per-sample weighting, clip, and a monitor worth reading

Everything above optimizes *what happens between* optimizer steps. This
addendum is about the steps themselves: a 2026-09-29 review of the
managed route against the legacy `core/` loop found the managed route
had silently dropped four stabilizers when it was written, which
together explain why trained LoRAs came out "slightly destructive" at
strength 1.0 even when the loss curve looked healthy.

- **`grad_accum` Port (default 1): effective batch was 2.** The legacy
  loop trained with `grad_accum=6`; the managed route had no
  accumulation at all, so every "step" was a single batch-2 update --
  noisier gradients and a different effective LR regime than the run the
  LR was tuned against. `steps` now counts *optimizer* steps: each step
  consumes K micro-steps (fresh batch, forward, backward) with the loss
  divided by K, and exactly one `zero_grad`/`update_lr`/`step()`/
  `on_step`/monitor report at the window's boundary. LR schedules key
  on the optimizer step, so warmup and decay see the same step count
  they would without accumulation. This is a deliberate divergence from
  legacy `core/train_step.py`, where `steps` counted micro-steps and the
  displayed step/total were batch-position counts -- keeping the old
  meaning would have made `steps` mean different things at K=1 and K>1.
- **`WarmupLRSchedule` (`nodes/train/schedule.py`): the legacy
  200-step warmup was never ported.** LR went straight to 1.5e-5 at
  step 0. The wrapper lerps from `warmup_start` (default 0.0) toward
  the *wrapped schedule's own value at the same step* -- so a cosine
  inside warmup already tracks the cosine, and frac reaches 1.0 exactly
  at the last warmup step (continuous join, no jump). `warmup_steps=0`
  is a passthrough; the node is registered in the graph.
- **Loss weighting is now applied per sample, on both routes.** The
  old code computed one scalar from the batch's *mean* sigma
  (`w(mean sigma) * mean(loss)`). For Min-SNR that weight is a
  nonlinear function of sigma, so on any mixed-t batch -- and batches
  carry per-sample t almost always -- the scalar disagreed with the
  intended `mean(w(sigma_i) * l_i)`, over-weighting or under-weighting
  every step depending on where the t-samples landed. Both
  `ManagedLoRATrainerNode`'s and `step_pipeline.py`'s `LossPhase` now
  weight per sample when sigma is per-sample (mean-sigma fallback for
  shared-sigma schedules, uniform is bit-identical either way).
- **`grad_clip_max_norm` Port (default 0.0): no gradient clipping
  anywhere, fused or not.** Clipping runs `clip_grad_norm_` once per
  optimizer step, on the boundary's full accumulated gradient only
  (never on interior micro-steps, where it would rescale a window in
  progress). It is a build-time `ValueError` on a fused optimizer --
  those optimizers fire their updates inside `backward()` hooks, so
  clipping after backward is already too late and pretending otherwise
  would be a silent no-op -- and for negative max-norm.

## Eighth addendum: `exact` joins the bucket-balance loop -- balance-steered t targeting

The actual intent was targeting *inside* the
bucket-balance toolkit: the balance shows which zone is underutilized
or losing badly, and you hit precise timesteps there. So the two now
compose:

- **`BucketBalance.exact_probs(t_low, t_high, values)`** (data side,
  next to `sampling_probs`): one draw weight per list entry = its
  bucket's `(current/baseline)^sample_bias` share -- the adaptive
  sampler's ratio semantics reused verbatim, so `sample_bias` is the
  steering-strength knob on both data-side modes, and mode-independence
  holds here too (the gradient side can be `off` while the data side
  steers). Duplicates in the list keep their multiplied share; a value
  outside the active coverage (impossible for range-validated
  `t_values`, defensive anyway) gets the mean of the active weights --
  neutral, never a fabricated preference. It deliberately does *not*
  set `_last_sample_range`, so `prob_t_*` (the adaptive *range*
  distribution) stays absent instead of misreporting a distribution
  over the range when the real draws are over the list.
- **`TrainTimeSampler._draw_exact`**: when the wired balance returns
  weights that actually differ, draws become weighted picks over the
  list (cursor set aside -- a pick has no position); when they're all
  equal, the plain pinned cycle runs unchanged. That "all-equal" line
  is the whole honesty contract: no balance wired, balance never
  observed, all buckets unwarmed, `sample_bias=0`, or every listed
  value in one bucket -- in every one of those cases the balance has no
  real opinion, and the user's equal-share pin is what they get. The
  cursor is per-loader as before, so the cycle still spans batches and
  epochs.
