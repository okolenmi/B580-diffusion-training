*[← docs/known-issues index](README.md)*

# Resolved

All five entries below were moved here from
[`pending-testing.md`](pending-testing.md) after being run on real
hardware on 2026-09-28 (Intel Arc B580, 12 GB, torch 2.12.1+xpu), via
`scripts/hw_validate.py` / `scripts/hw_validation_batch.sh` -- each
entry's "Confirmed" paragraph carries the measured result. (The topmost
two entries are a different provenance: 2026-09-29's is a review-caught
bug, fixed in the encoder rather than in the test that caught it, and
the one below it a user-reported wrong number in this project's own
documentation, investigated and corrected on hardware the same day.)

- **[2026-09-29] A cold `CachingTextEncoder.encode()` called
  `ensure_loaded()` twice (once per cache half), and the test that
  caught it was edited to expect two instead of fixing the encoder.**
  The split-key redesign gave each half (`encode_prompt_only` /
  `resolution_embedding`) its own `ensure_loaded()`, so a both-cold
  encode fired it once per half; `c703aa6` changed
  `check_resource_control_called_only_on_miss`'s assertion from one
  call to two, justified by "the second call is a no-op in real
  ResourceControlHandle (already resident)" -- not true:
  `BudgetedResourceControlHandle.ensure_loaded()` skips the reload when
  resident but unconditionally runs `_make_room()`, whose first act is a
  `memory_stats()` read -- a real device query on every call. Every
  both-halves-cold encode paid that query twice for the rest of the run
  with caching on. Fixed in the encoder this time:
  `CachingTextEncoder.encode()` checks both cache keys before either
  half loads and calls `ensure_loaded()` once, suppressing the halves'
  own per-miss calls only for the duration of that call (direct half
  calls keep theirs); the test asserts one again, plus a new check that
  direct half calls still ensure. The docstrings now state the real
  cost too: `ensure_loaded()`'s no longer calls the resident-path
  check just "cheap" -- it says the reload branch is skipped yet
  `_make_room()`'s `memory_stats()` read (a real device query) still
  runs every call.

- **[2026-09-28] User-reported wrong VRAM figure in this project's docs
  ("reserved flat at ~6034 MB" for the legacy health check) -- the
  report was right: real consumption was ~11.4/12.2 GB, and the
  misleading number was a snapshot-ordering artifact, now fixed.**
  Investigation (legacy `convert.py`, `runs/hw_validation/legacy_check.toml`,
  instrumented phase snapshots + tensor census + a cross-process
  `torch.xpu.mem_get_info` monitor): steady state during training is
  **reserved 9942 MB, live tensors ("allocated" and an independent
  `gc`-based census, matching to 0.6 MB) 5624 MB, external driver total
  ~11406 MB** -- which reconciles exactly (allocator free-block pool
  ~4.3 GB + ~900 MB desktop baseline + context overhead), and matches
  both the user's own monitoring (11.4/11.9 GB) and the cross-process
  query. The ~6034 MB in the docs came from `core/train_step.py`'s
  maintenance block running `xpu_empty_cache()` *before* the
  `[vram]` snapshot every 250 micro-steps: `empty_cache()` collapses
  reserved to roughly allocated, so every snapshot the docs quoted was
  a post-free reading ~3.9 GB below steady state. Fix: the block now
  snapshots before the maintenance (labelled `pre-maintenance`) and
  again after (labelled `post-empty_cache`), so the reported reserved
  is the real device-relevant number and the drop is still observable.
  Ruled out along the way (each measured, not assumed): torch's
  allocator stats and the census agree everywhere tested (no
  under-counting by torch); the external reading is genuine device
  memory (host-RAM and pinned-memory injections don't move it);
  the earlier "5.6 GB hidden outside torch" reading was entirely the
  post-`empty_cache` artifact. Also cross-checked that the numbers in
  this file's `pending-testing` entries were *not* affected: those are
  `peak_reserved` values, and an independent driver-level rerun of
  `A_after` (`A_after_mon`) peaked at 10140 MB external ≈ 8592 MB
  reserved + ~950 MB desktop + overhead -- consistent. Side finding,
  tracked in [`deferred.md`](deferred.md): `torch.xpu`'s
  `pin_memory()` does not actually lock pages on this build.

- **[2026-08, confirmed 2026-09-28] VRAM ratchet on non-square
  datasets -- `max_aspect_ratio` cap confirmed bounded on real
  hardware.** Original diagnosis (unchanged, at the time of the move
  this was the newest of the five and the only one whose fix predated
  the run): the hypothesis that this was allocator fragmentation tested
  wrong (`num_alloc_retries` 0, flat on uniform datasets), and real
  per-phase capture found the actual mechanism -- `resize_mode="fit"`
  with no cap on the long side pushed genuinely larger tensors through
  exactly one phase boundary (`forward`), the allocator grabbed a
  bigger reserved block, and kept it permanently (+560 MB in one step).
  Fix: `manager/builder.py`'s `run_lora_ingestion_task`'s
  `max_aspect_ratio` parameter (default `2.0`) splits over-long
  "fit"-resized images into multiple same-caption crops, wired through
  the ingestion UI. **Confirmed:** `datasets/non-square` was
  re-ingested with the cap (its `sources.config` records
  `"max_aspect_ratio": 1.5`), then a 60-step main-route run on it
  (label `B_ratchet`, genuinely variable resolutions -- latents from
  48x64 to 96x64, both orientations, 273 samples) came back bounded:
  per-step reserved first 8104 MB, last 8168 MB, total drift 64 MB,
  **max single-step jump +40 MB** (vs the pre-fix +560 MB permanent
  jump). No ratchet.

- **[2026-09, confirmed 2026-09-28] attention-block checkpointing
  (`nodes/model/attention_checkpointing.py`) -- before/after on real
  hardware, and the "before" is an outright OOM.** The fix makes
  `use_checkpoint=True` actually reach SDXL's
  `BasicTransformerBlock` stacks (ComfyUI's
  `BasicTransformerBlock.__init__` never assigns its `checkpoint`
  argument, so only `ResBlock` was ever checkpointed -- see the
  original entry's root-cause work in git history for this file's
  pre-move version). **Confirmed:** two identical main-route runs on
  the uniform 1024-dataset (40 steps, rank 64, budget 11500), differing
  only in whether `enable_attention_block_checkpointing()` was
  active: with the patch (`A_after`) -- 40/40 steps, per-step peak
  reserved **8592 MB**, 0.943 steps/sec; with it neutered to
  reproduce the pre-fix ResBlock-only behavior (`A_before`) -- hard
  `torch.OutOfMemoryError` on the **first forward pass** (10.76 GiB
  allocated of 11.93 GiB total, zero steps completed). So this wasn't
  a marginal activation reduction: 1024² at batch 1 is simply not
  trainable on this 12 GB card without it, which is the same
  activation-dominance shape as the two OOM reports that motivated
  the fix.

- **[2026-09-20, confirmed 2026-09-28] `BudgetedResourceControlHandle`'s
  `synchronize()` / `strict` / `release()` hardening -- exercised under
  real VRAM pressure, both outcomes as designed.** (Original entry:
  defensive hardening motivated by the open "Device lost" report,
  previously verified only against a scripted fake `DeviceContext` in
  `smoke_test_resource_control_strict.py`.) **Confirmed:** two
  identical main-route runs at `vram_budget_mb=2500` against actual
  ~8114 MB usage on the `1image` dataset: non-strict (`C_pressure`)
  -- 30/30 steps, no hang, no device-lost, the offload path taken
  every step (allocated dropped 7886 → 6325 MB after the first offload
  and held there -- text encoder offloaded, allocator kept its
  reservation, exactly the accounted-for behavior); strict
  (`C_strict`) -- raised precisely as designed: *"7538MB reserved
  still exceeds the 1988MB usable budget (2500MB minus 512MB reserve)
  after offloading every resident registered as offloadable"*. The
  same run also exercised `SDXLTextEncoder.offload()`/`reload()`
  under real pressure every step (via the caching wrapper), covering
  that companion fix from the same session. **Scope caveat kept from
  the original entry:** this exercises the `nodes/` rewrite's offload
  path, not the legacy `core/trainer.py` path the open "Device lost"
  report is about -- that relationship stays unconfirmed (see
  [`open.md`](open.md)).

- **[2026-09-20, confirmed 2026-09-28] `AdaptiveResidencyController`
  perf regression (managed route ~4.7x slower than main) -- fixed on
  the hardware that produced the report.** Original report: 0.36 vs
  ~1.7 steps/sec (managed vs main), against a stated 12500 MB budget
  where offloading was never necessary. **Confirmed:** post-fix, two
  identical runs (uniform 1024-dataset, 40 steps, rank 64, budget
  11500, AdamW, attention checkpointing on) differing only in trainer
  node: managed (`D_managed`) **0.897 steps/sec** vs main
  (`A_after`) **0.943 steps/sec** -- managed at 95% of main-route
  speed (was ~21%), identical per-step peak reserved (8592 MB both).
  The residency controller calibrated (measured peak 8592 MB over 3
  calibration steps, usable 9889 MB) and correctly stayed fully
  resident. The still-open costs listed in
  `docs/design/resources-controller/09-trainer-integration-and-vram-safety.md`
  (no text-encoder caching on this route, no pinned host memory) are
  now the measured ~5% gap's plausible homes, not a 4.7x mystery.

- **[2026-09-20, confirmed 2026-09-28] `AdaptiveResidencyController`'s
  ongoing escalation + `residency_safety_margin` -- the OOM on the
  variable-resolution dataset is gone.** Original report: calibration
  sampled only smaller images, "stay resident" locked in before the
  worst case was measured, and a run then OOM'd at 10218 MB reserved
  on `datasets/non-square`. **Confirmed:** post-fix run on that same
  dataset (`E_managed_nonsq`, 60 steps, budget 8000 MB -- i.e. usable
  6739 MB after reserve + 10% margin, deliberately below what the run
  would use): calibration measured 8036 MB > usable, controller
  escalated ("releasing optimizer, text_encoder"), and the run
  finished **60/60 steps with no OOM**, per-step peak max 8036 MB.
  Escalation's "next occurrence, not this one" limit was acceptable
  in practice across 60 genuinely variable-resolution steps. The run
  also put a number on the *disclosed, still-open* limit: releasing
  residents could not bring reserved anywhere near the 6739 MB budget
  (model 4897 MB + activations dominate; reserved settled ~8000 MB),
  confirming that release-based escalation cannot budget-fit an
  activation-dominant run -- the lever for that remains activation
  checkpointing (the attention-block entry above is now wired and
  confirmed; full activation management is still the backlog item).

- **[2026-09] `core.optimizers.ChunkedXPUAdafactor`/`FusedXPUAdafactor`
  silently corrupted their own momentum buffer for float32 parameters
  with `beta1` (momentum) set.** `g = self.exp_avg[i]` aliased the
  momentum buffer (no copy); the following
  `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))` called
  `.to(dtype=p.dtype)`, which for a float32 parameter (state is already
  float32) returned the *same object*, not a copy -- so the subsequent
  `.mul_(alpha_t)` mutated the momentum buffer in place. Net effect:
  every step, right after using the momentum buffer to compute that
  step's update, the buffer got permanently shrunk by `alpha_t` (~lr)
  as an unintended side effect. Confirmed directly, not theorized --
  see `nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py`'s
  `check_legacy_float32_momentum_no_longer_corrupted()` for the
  `FusedXPUAdafactor` case; `ChunkedXPUAdafactor`'s copy of the same
  pattern (`core/optimizers.py`, main-parameter path) confirmed the
  identical way by reading it directly. bf16 parameters were never
  affected (`.to(dtype=p.dtype)` performs a real cast there, producing a
  genuine copy).
  **Scope differed between the two, checked precisely rather than
  assumed:** `FusedXPUAdafactor`'s buggy line was shared by both its
  tiny-parameter (< 10,000 element) and main-parameter code paths, so it
  affected every parameter size. `ChunkedXPUAdafactor`'s sat only in its
  main-parameter path (guarded by `p.numel() < 10_000` routing tiny
  parameters elsewhere instead) -- its own tiny-parameter path applies
  updates via a different mechanism (`ws[s:e].add_(..., alpha=-alpha_t)`,
  no aliasing problem) and was never affected. `ForeachXPUAdafactor`
  never had this bug at all -- both its factored and unfactored paths
  use non-in-place `.mul(alpha_t)`, which always returns a new tensor
  regardless of dtype, never aliasing the momentum buffer.
  `nodes/optimizer/algorithms/adafactor.py`'s `AdafactorAlgorithm` never
  had this bug either way. Fix: forced a real copy
  (`.to(dtype=p.dtype, copy=True)`) in both `core/optimizers.py`
  locations, regardless of whether the dtype conversion alone would
  already produce one -- `core/` is untouched by the `nodes/` rewrite's
  own restructuring (see `docs/architecture.md`), but bugs found in it
  get fixed in place, which this is. Since the fix makes legacy's
  momentum math structurally identical to `AdafactorAlgorithm`'s (both:
  blend into the buffer, copy, scale, cast), the float32+momentum
  combination that `smoke_test_fused_adafactor_equivalence.py` used to
  deliberately exclude from its equivalence grid (because the bug made
  it meaningless to compare against) is included now too -- both the
  fix itself and the newly-included float32+momentum equivalence checks
  confirmed fixed by user (both tests passed).

- **[2026-09] `DeviceResident.footprint_bytes()` didn't check actual
  device placement.** `footprint_bytes()` is documented as "current
  device-memory usage," but every implementation just summed
  `numel() * element_size()` over whatever tensors it held -- a number
  that doesn't change when a tensor moves from GPU to CPU, since
  shape/dtype don't change either. Practical consequence: after
  `offload()`, `footprint_bytes()` kept reporting the same pre-offload
  total, as if nothing had moved. This directly undermined a real,
  designed monitoring feature -- `nodes/train/step_pipeline.py`'s own
  comment describes comparing `ResourceCoordinator.total_footprint_bytes()`'s
  sum against the real device-driver-reported VRAM stats as "a useful
  sanity signal," with "a growing gap between them worth investigating"
  -- offloading anything would have produced exactly that gap, not
  because of a real accounting problem but because `footprint_bytes()`
  itself couldn't tell CPU-resident from device-resident. Confirmed by
  reading all four concrete `DeviceResident` implementations directly,
  not assumed to generalize from one: `SDXLTextEncoder`
  (`nodes/model/text_encoder.py`), `ComfyUNetTrainableModel`
  (`nodes/model/lora_injector.py`), `ComposedOptimizerHandle`
  (`nodes/optimizer/composed.py`, also covers
  `ComposedFusedOptimizerHandle` via inheritance), and
  `AdafactorOptimizerHandle` (`nodes/optimizer/adafactor.py`, the one
  remaining legacy-wrapping optimizer) -- all four had the identical
  gap. `nodes/model/text_encoder_cache.py`'s caching wrapper delegates
  to its inner resident's `footprint_bytes()`, so needed no separate
  fix. Fix: each implementation now tracks whether it's currently
  offloaded (`SDXLTextEncoder`/`ComfyUNetTrainableModel` already tracked
  `self._device_before_offload` for `reload()`'s own use, just never
  checked it here; `ComposedOptimizerHandle`/`AdafactorOptimizerHandle`
  needed a new `self._offloaded` flag) and return `0` from
  `footprint_bytes()` while it's set -- correct, not a compromise: `0`
  bytes of *device* memory is exactly right for something living in
  host RAM. Second, related bug caught while implementing this:
  `reload()` in both `SDXLTextEncoder` and `ComfyUNetTrainableModel`
  read `_device_before_offload` but never reset it afterward -- harmless
  before (nothing else read it), but would have made `footprint_bytes()`
  incorrectly report `0` forever after the *first* offload, even once
  reloaded, without also fixing this. Confirmed fixed by user (both
  tests passed) via
  `nodes/smoke_tests/smoke_test_footprint_bytes_after_offload.py`,
  covering all four implementations.

- **[2026-09] `AdafactorOptimizerNode`'s legacy-wrapping siblings
  (`ForeachAdafactorOptimizerNode`, `FusedAdafactorOptimizerNode`) had a
  real, unreplicated small-parameter (< 10,000 element) code path
  relative to their `Composed*` equivalents -- confirmed and closed on
  real torch, not just read from source.** `ChunkedXPUAdafactor`/
  `FusedXPUAdafactor` both had a tiny-parameter fast path (a plain
  elementwise second-moment EMA in place of the row/col factored
  approximation) that `AdafactorAlgorithm` didn't cover -- and the two
  didn't even agree with each other on the mechanism
  (`FusedXPUAdafactor`'s is genuinely per-parameter;
  `ChunkedXPUAdafactor`'s ties every tiny parameter in the whole
  optimizer together into one shared clip and EMA state, a
  cross-parameter batching concern, not a per-parameter algorithm one).
  `ForeachXPUAdafactor` turned out to have no tiny-parameter special
  case at all. Confirmed by a real-torch run
  (`nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`):
  Foreach came back equivalent to `ComposedAdafactorOptimizerNode(
  strategy="foreach")` at floating-point-noise magnitude (no algorithm
  change needed) -- `foreach_adafactor.py`/`ForeachAdafactorOptimizerNode`
  deleted. Fused's gap was real (1e-3 to 1e-2 magnitude, not noise);
  closed by adding an opt-in `tiny_parameter_threshold` to
  `AdafactorAlgorithm` (used only by `ComposedFusedAdafactorOptimizerNode`,
  deliberately not by the chunked/foreach/simple/shape_grouped strategies,
  since that would have broken the just-confirmed Foreach match) --
  confirmed closed on real torch, within this project's own
  already-established equivalence tolerances for this pair (`1e-4`
  float32, `1e-2` bf16) -- `fused_adafactor.py`/`FusedAdafactorOptimizerNode`
  deleted too. `ChunkedXPUAdafactor`'s cross-parameter-batching version
  is a different, bigger problem (new `ExecutionStrategy`-level
  machinery, not an algorithm change) and is not resolved --
  `AdafactorOptimizerNode` stays registered for it, tracked as real
  future work in `docs/design/09-prioritized-backlog.md`, not carried
  here as an open bug since nothing is broken, there's just a capability
  gap. Permanent regression coverage:
  `nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py`
  (tiny-parameter case) and
  `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py` Part A
  (Foreach case).

- **[2026-08] 8 real, working Node classes existed but weren't
  selectable in the graph editor -- `server/nodegraph_registry.py`'s
  list was stale.** Confirmed directly, not a hypothesis: walked every
  concrete `Node` subclass under `nodes/` and diffed against
  `server.nodegraph_registry.get_registry()`'s actual returned set.
  Missing: `P2LossWeightingNode`, `PrefetchingBatchSourceNode`, and six
  `Composed*OptimizerNode` classes predating the 2026-08 `nodes/`
  session entirely. Fix: added each to `server/nodegraph_registry.py`'s
  import list and `classes` list. Confirmed fixed by user ("I can now
  see and use new nodes").

- **[2026-07] CAME optimizer VRAM near-ceiling hang after ~60 steps.** Root
  cause: `res` and `update` in `ChunkedXPUCAME.step()` each allocated a fresh
  full-parameter-sized tensor per step (on top of Adafactor's baseline
  scratch-buffer usage), slowly fragmenting VRAM near the ceiling. Fixed by
  reusing the existing scratch buffer in place for both. Confirmed fixed by
  user.

- **[2026-07] Default `snr_weighting: "snr"` used the v-prediction Min-SNR
  formula (`snr/(snr+1)`) unconditionally, including for the default
  `student_type: "eps"`.** For eps-prediction the correct uncapped form is
  trivially 1.0 (uniform); the old default gave ~99% weight to easy/low-noise
  steps and ~1% to high-noise/structural steps -- close to the opposite of
  what's wanted. Fixed by branching `snr`/`min_snr_5`/`decay_snr` on
  `student_type`. Recommended switching configs to `min_snr_5` explicitly
  (the correctly-implemented standard choice for eps) rather than relying on
  `snr` reducing to a uniform no-op.

- **[2026-07] `grad_accum` inflated "steps" to mean micro-batches, not real
  optimizer updates.** `steps: 5000, grad_accum: 32` only did `5000/32 = 156`
  real weight updates; LR schedule, save/preview cadence, and the dashboard
  all silently used the wrong count. No warning, and the shipped example
  config (`convert-cfg.toml`) already had `grad_accum: 32`. Refactored so
  `steps` means real optimizer updates everywhere (dashboard, saves,
  previews, LR schedule); cache size and micro-batch loop scale internally by
  `grad_accum` instead. Confirmed working by user (correct step count,
  expected per-step timing).
