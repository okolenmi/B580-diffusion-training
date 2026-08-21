# Suspicious findings / deferred work

**Note (2026-08): this file predates and sits outside the `nodes/`
design-doc effort** (see `docs/training_pipeline_design.md`). Unless a
`nodes/` path is named explicitly, an entry describes the legacy `core/`
production pipeline, not the design doc's rewrite. Treat this as an
informal, unaudited list, not a spec -- entries can be stale, already
fixed elsewhere, or (as two entries here used to be) about a feature
`nodes/` never had in the first place: `nodes/` currently implements
plain supervised LoRA training only, no distillation and no chain-mixing
of any kind, so a DAgger/chain-mixing finding about `core/trainer.py`
isn't a `nodes/` backlog item and was removed rather than carried
forward. Two dangling pointers to `docs/optimizer_execution_redesign_plan.md`
and `docs/nodes_package_design.md` (both deleted) were also cleaned up
below -- the substantive content they pointed from is kept, just not the
broken link.

Running list of things noticed during review that aren't confirmed bugs (or
are confirmed but not urgent) so they don't get lost. Newest first.

## Open

- **[2026-08] `DeviceResident.footprint_bytes()` doesn't check actual
  device placement anywhere.** Surfaced while investigating a VRAM
  report later found to have a different, unrelated root cause (see
  "Pending user testing" below): `tracked_footprint_mb` was consistently
  reported *higher* than `vram_allocated_mb` in real profiling output,
  backwards from what static-weight-only accounting vs. real
  allocated-including-activations should show. Not traced further --
  still open, independent of the VRAM-ratchet finding below (that one's
  fully explained by the allocator's `reserved` behavior on an uncapped
  image long side, doesn't touch `footprint_bytes()` at all).

- **[2026-07] "Device lost" errors and silent training hangs after
  VRAM-pressure events, reported from real ComfyUI use (legacy `core/`
  pipeline, not `nodes/`).** User-reported, not yet investigated here.
  Symptom: not a normal OOM -- either a device-lost error or a silent
  hang, most reliably reproduced by a VRAM-heavy sequence (merging three
  6GB models, generating with an intermediate merge state, then
  generating again with a different base model) and, separately, seen in
  this project itself after a VRAM spike during preview generation's VAE
  decode step -- loss of a few hundred MB stabilizes (frees back down)
  but training hangs a few steps later *despite* free VRAM being
  available afterward. User's own read, worth taking seriously: something
  gets offloaded under memory pressure but isn't correctly loaded back,
  even though there's room for it. Likely related to the "Persistent
  ~500MB VRAM growth after preview generation" entry below (same VAE
  decode trigger point) but the *symptom* here (hang/device-lost, not
  just VRAM not dropping back down) is a distinct, arguably more serious
  report -- not confirmed to be the same root cause, not assumed to be
  either. A `kohya-ss/musubi-tuner` discussion training Wan2.2 on the
  same B580 hardware describes a matching hang-after-offload symptom,
  traced there to a `synchronize_device()` call missing its `device`
  argument on the non-CUDA path -- a plausible root-cause *shape* (async/
  non-blocking transfer without a matching explicit synchronize on the
  XPU path) worth checking `core/trainer.py`'s own offload code for, not
  a confirmed diagnosis here. Not investigated further this session --
  out of scope for `nodes/`-only work, and needs `core/trainer.py`,
  which `nodes/` doesn't touch.

## Resolved

- **[2026-08] A dataset smaller than `batch_size` (real user report: 1
  image, `batch_size=2`) silently produced zero batches forever, then
  crashed training several frames away with a bare, unexplained
  `StopIteration`.** `manager/loader.py`'s `ManagedDatasetLoader.__iter__()`
  correctly drops each bucket's incomplete last chunk when
  `shuffle=True` (intentional, common practice) -- but when a bucket's
  samples are *entirely* one incomplete chunk (the whole dataset, or
  every per-(prompt, size) bucket, is smaller than `batch_size`),
  dropping "the last incomplete chunk" drops everything, silently,
  every epoch. Reproduced directly against the real class with a real
  1-row sqlite dataset (`shuffle=True, batch_size=2`): `len(loader)`
  reported `1` (misleading), `list(loader)` yielded zero batches. Traced
  the actual crash it causes: `nodes/train/step_pipeline.py`'s
  `FetchBatchPhase` catches exactly one `StopIteration` to wrap to a new
  epoch, then hits an uncaught second `StopIteration` immediately after
  (the fresh epoch is just as empty) -- that's what actually reaches the
  user, several frames removed from the real cause and with no
  indication why. Fixed by raising a clear, specific `ValueError` at the
  actual source (`__iter__()`) instead of silently returning an empty
  generator -- a genuinely empty dataset (0 samples) is unaffected,
  still yields nothing, not an error; only the "had samples, all of them
  got dropped" case now raises. Regression-tested against the real
  class via the same real sqlite+safetensors harness
  `smoke_test_lora_raw_dataset.py` already established (no mocks).

  **The identical bug pattern exists in `core/cache_utils.py`'s
  `shuffle_and_rebatch_cache()`** (same "drop an incomplete last chunk,
  no check for the whole batch ending up empty" shape, confirmed by
  reading it directly) -- not fixed, on purpose: that file is
  `core/`-tier, reference-only, unreachable from `nodes/` (confirmed --
  only `core/cache_trajectory.py`/`core/cache_random.py` call it, both
  part of the legacy caching flow `core/trainer.py` uses, not anything
  `SupervisedLoRATrainerNode` touches). Flagged here so it isn't
  forgotten, not silently left for someone to rediscover the hard way if
  that legacy path's own small-dataset case ever gets hit for real.

- **[2026-08] `Algorithm.compute_update_batched()`'s default fallback
  silently applied the wrong `decay` to some group members when `decay`
  genuinely varied across a group -- found while building
  `ShapeGroupedForeachStrategy`, not specific to it.** The fallback (used
  by any `Algorithm` without its own batched override, and by
  `AdafactorAlgorithm.compute_update_batched()`'s own
  `scale_parameter=True` case) looped over group members computing
  `compute_update()` for each, but kept only the *last* member's `decay`
  in a plain variable that got overwritten each iteration -- correct
  only when `decay` is actually uniform across the group, which the
  method's own docstring already claimed but never checked.
  `AdafactorAlgorithm`'s `scale_parameter=True` breaks that assumption
  for real: `alpha_t` (and therefore `decay = 1 - wd*alpha_t`) depends on
  each parameter's own live norm, genuinely different per member. An
  earlier equivalence test of this exact fallback path
  (`smoke_test_adafactor_shape_grouped_equivalence.py`'s
  `scale_parameter=True` check) happened to use `weight_decay=0.0`,
  where `decay` is always `None` regardless of `alpha_t` -- silently
  sidestepping the bug rather than proving its absence. Surfaced by a
  real equivalence-test failure once `weight_decay != 0` was actually
  run through a batched strategy while building
  `smoke_test_shape_grouped_foreach_equivalence.py`. Fixed to detect a
  varying `decay` and raise a clear, specific `RuntimeError` naming the
  problem and what to do instead (use `simple`/`chunked`/`foreach`, or
  give the `Algorithm` a real per-member-decay-aware override) rather
  than silently corrupting training for every group member but the last.
  Both the new strategy's test and the existing Adafactor test now cover
  this combination directly.

- **[2026-08] `ComposedAdafactorOptimizerNode`/`ComposedAdamWOptimizerNode`
  crashed real training with `SupervisedLoRATrainerNode: ZeroDivisionError:
  division by zero` -- real user report on real hardware, not a
  hypothesis.** Traced directly, not guessed: the crash was reported with
  `strategy="foreach"` (`AdafactorAlgorithm+ForeachApplyStrategy`, per the
  server console's own printed optimizer id), which calls
  `AdafactorAlgorithm.compute_update()` once per parameter -- confirmed by
  reading `ForeachApplyStrategy` directly, it only batches the final
  apply step, never the per-parameter algorithm math. Root cause in
  `compute_update()` itself: `clip_mul = min(1.0, self.clip_threshold /
  float(rms_g))` converts `rms_g` (a tensor) to a Python float *before*
  dividing -- when a real parameter's gradient this step has an exactly
  zero norm (confirmed reachable, not theoretical -- reproduced directly
  with a plain zero-gradient tensor), Python float division by exactly
  0.0 raises `ZeroDivisionError`. `core.optimizers.ChunkedXPUAdafactor`
  (the legacy class this was ported from) does the identical clip
  computation but stays in tensor space the whole time
  (`torch.clamp(self.clip_threshold / rms_g, max=1.0)`), which produces
  `inf` then clamps to `1.0` -- never raises, confirmed directly by
  reading it. Fix: match the legacy tensor-space computation exactly,
  converting to Python float only after clamping to a finite value.
  `CAMEAlgorithm` doesn't have this bug -- confirmed by reading it, its
  own clip computation divides `rms` by a fixed nonzero `clip_threshold`
  constant, not by a potentially-zero tensor, a different formula
  structure. The random-Gaussian-gradient equivalence tests already
  covering `AdafactorAlgorithm` structurally could never have caught
  this (`torch.randn()` essentially never produces an exact `0.0` norm)
  -- a dedicated zero-gradient regression check was added specifically
  to close that coverage gap, not just to confirm this one fix.
  Confirmed bit-exact against the legacy reference's own handling of a
  zero gradient. Awaiting confirmation the fix resolves the real crash
  on the user's own hardware/workflow.

- **[2026-08] `ComposedAdafactorOptimizerNode`/`ComposedAdamWOptimizerNode`
  didn't accept `strategy="shape_grouped"` at all --
  `ValueError: Unknown strategy 'shape_grouped' -- choose one of
  ['simple', 'chunked', 'foreach']`, real user report.** `ShapeGroupedBatchStrategy`
  itself was real and already registered on `ComposedCAMEOptimizerNode`,
  but the same registration was simply missing from the other two
  composed nodes' own `_STRATEGIES` dicts -- confirmed by reading all
  three directly, not assumed from the error message alone. Also added a
  real `AdamWAlgorithm.compute_update_batched()` while fixing this
  (AdamW's math has no factored reduction and no clip-based division to
  worry about, so this was a small, low-risk addition once CAME's and
  Adafactor's own batched overrides had already established the
  pattern) -- see `smoke_test_adamw_shape_grouped_equivalence.py`
  (bit-exact against the per-member reference; the existing
  `smoke_test_adamw_equivalence.py` technically already exercised
  `shape_grouped` once registered, but its own two parameters are
  different shapes, so it only ever hit the singleton-group fallback,
  never the real batched path -- this is why a dedicated test with real
  same-shape groups was needed).

  **The first fix was wrong, called out directly and correctly by the
  user: three byte-identical copies of `_STRATEGIES` plus its dispatch
  logic plus its doc string, one per composed node, is exactly the
  structure that produces this bug class -- fixing two of three copies
  and leaving the pattern in place would just leave it ready to happen
  again.** Confirmed the duplication was real and worse than just the
  dict: the `strategy` Port's own doc string (hand-written plain text
  listing valid names) was `composed_adafactor.py`/`composed_adamw.py`'s
  *own separate* copy, and updating their dicts to add `shape_grouped`
  did not update those doc strings -- they still said "One of 'simple',
  'chunked', 'foreach'" after the first fix, a live second bug of the
  identical class, introduced while fixing the first one.
  `composed_came.py`'s own doc string happened to already be correct,
  not because the duplication was safe but because nobody had touched it
  since `shape_grouped` was first added there. Real fix:
  `nodes/optimizer/strategy_registry.py` -- one `STRATEGIES` dict, one
  `resolve_strategy()` dispatch function, one doc string generated from
  the dict itself (not hand-written), imported by all three composed
  nodes. Both bug classes (a strategy missing from some copies, a doc
  string out of sync with the dict) are now structurally impossible, not
  just fixed once -- there is nothing left to independently drift.

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

## Deferred (not urgent, revisit later)

- **[2026-07] New finding, informational only:
  `ChunkedXPUAdafactor`'s momentum handling corrupts `exp_avg` in place
  when a parameter's dtype is float32.** Found while building and
  verifying `nodes/optimizer/algorithms/adafactor.py`'s `AdafactorAlgorithm`
  against this class directly. In `step()`: `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))`, where
  `g` is `self.exp_avg[i]` a few lines above (aliased, not copied). When
  `p.dtype == torch.float32` (same as the internal state dtype),
  `.to(dtype=p.dtype)` is a documented no-op returning the *same tensor
  object* -- confirmed directly (`t.to(dtype=t.dtype) is t` -> `True`) --
  so the following `.mul_(alpha_t)` permanently shrinks the momentum
  buffer itself by `alpha_t` (~`lr`) every step, rather than only scaling
  a throwaway copy for the parameter update. **Does not affect real
  training**: this codebase trains in bf16, and `.to(dtype=bf16)` from a
  float32 buffer always allocates a fresh tensor, so the aliasing (and
  therefore the corruption) never happens in practice -- confirmed by
  re-running the same comparison under bf16 and seeing the divergence
  collapse to ordinary quantization noise, no larger than the
  no-momentum case's own bf16 noise. Left here as a record, not fixed --
  `nodes/` doesn't touch `core/optimizers.py`, and there's no evidence
  this has ever caused a real-training problem to chase.

- **[2026-07] Note for future sessions: `nodes/memory/manager.py`'s new
  `MemoryManager` structurally prevents the reset-vs-free asymmetry bug
  class behind the "CAME optimizer VRAM near-ceiling hang" entry above,
  for anything built through `nodes/` going forward.** This does **not**
  fix or touch `core/optimizers.py`'s legacy classes -- per `nodes/`'s
  existing rule, that file hasn't been modified. Left here as a pointer,
  not a claim of resolution: once the node-graph optimizer path replaces
  the legacy one, this whole class of VRAM-lifecycle bug should stop
  being something to watch for by construction, rather than something to
  keep re-auditing by hand.

- **CAME's tiny-param batching fast path.** `ChunkedXPUAdafactor` has a
  vectorized/batched fast path for many small parameters (relevant for LoRA's
  many small A/B matrices) that `ChunkedXPUCAME` doesn't replicate --
  deliberately deferred to keep the initial port reviewable. CAME has more
  per-parameter state than Adafactor (two factored row/col pairs instead of
  one, plus the momentum buffer), making the batching trick a real port, not
  a copy-paste. Only worth doing if CAME's per-step Python-loop overhead is
  actually a measured bottleneck for typical LoRA parameter counts.

- **CAME momentum buffer in bf16.** `exp_avg` is CAME's one genuinely new
  full-size buffer vs. Adafactor. Storing it in bf16 instead of fp32 would
  roughly halve that buffer's footprint at a small, untested precision cost
  on a smoothed EMA quantity. Shelved because the buffer-reuse fix above
  already resolved the near-ceiling hang on its own -- revisit only if VRAM
  is tight again after that fix.

- **`lora.py` legacy 2816->3072 padding path.** Fragile, hardcoded special
  case for loading old-format LoRA checkpoints. Now at least logs a warning
  when it fires (visibility fix already shipped). Generalizing it or removing
  it once no one has 2816-dim checkpoints left to load is lower priority.

- **[2026-08] No shared `MemoryManager` reachable from
  `SupervisedLoRATrainerNode.build()`.** Found while wiring
  `ResourceProfile` (`nodes/memory/profile.py`, design doc section 5.5):
  `nodes/optimizer/strategies/chunked.py`'s `ChunkedScratchBufferStrategy`
  constructs its own private `MemoryManager` when none is injected, and
  nothing between it and the trainer node passes one in, so there's no
  single instance to hand `ResourceProfile.capture()` --
  `memory_manager_stats` is `None` in every real run today. Harmless
  right now (each strategy's private manager is internally consistent on
  its own), not fixed here -- see `docs/training_pipeline_design.md`
  section 10's note on this for when it'd actually start to matter.

- **[2026-08] Only `ResBlock` instances ever route through this
  project's checkpoint patch -- attention blocks don't, in this pinned
  ComfyUI version.** Found while building `nodes/model/block_profiler.py`
  (design doc section 2.3, backlog item 1): cloned
  comfyanonymous/ComfyUI directly to confirm what `ctx.run_function`
  actually is. `comfy/ldm/modules/diffusionmodules/openaimodel.py`'s
  `ResBlock.forward()` calls `checkpoint(self._forward, ...)` for real;
  `comfy/ldm/modules/attention.py`'s `BasicTransformerBlock.forward()`
  does not call `checkpoint()` anywhere in its body, despite taking a
  `checkpoint=True` constructor parameter that looks like it should.
  Not a bug in this project -- `enable_frozen_param_safe_checkpointing()`
  only ever patches what ComfyUI itself routes through it -- but it does
  mean `GreedyRatioPlacement` (once wired in) can only ever place
  ResBlocks, and `use_checkpoint=True`'s real VRAM savings today only
  ever come from ResBlocks, never attention blocks, regardless of what
  the constructor parameter's name suggests.

- **[2026-08] DoRA's `magnitude` parameter isn't wired into checkpoint
  save/load.** `nodes/model/dora_layer.py`'s `DoRALinear`/`DoRAConv2d`
  are real and trainable (via `DoRAAdapter`, live-wired through
  `adapter_strategy_scope`), but `nodes/model/lora_saver.py` and
  `LoRACheckpointSaverNode`/`LoRACheckpointLoaderNode` only know about
  `lora_A`/`lora_B` -- a DoRA-trained `magnitude` (independently trained
  state, not derivable from `lora_A`/`lora_B` alone) would silently not
  get saved. `DoRALinear.load_dora_weights()` exists for the real
  round-trip once save-side wiring exists; nothing currently calls it.
  Not fixed here -- see `docs/training_pipeline_design.md` section 10.

- **`config_model.py` doesn't yet warn about grad_accum's real-update math
  anywhere in the UI/docs.** The step-counting refactor fixed the mechanism,
  but nothing explains "steps now means real updates, cache/compute cost
  scales with steps*grad_accum" to a new user reading the config file cold.

## Pending user testing

- **[2026-08] LoRA timestep gate now wired -- candidate fix for a real
  deformation/quality report.** User-reported after a real training run
  (rank 48, alpha 1, dropout 0, weight_decay 0, **t range [150, 999]**,
  LR 1e-5, clip_threshold 1, 18000 steps): at LoRA power 2.0 a visible,
  useful change came with significant anatomy/content deformation; at
  power 1.0 the useful change was much less visible but some deformation
  was still present.

  **Strong candidate, not confirmed**: `t range [150, 999]` excludes the
  low end of the timestep range (t<150). Separately confirmed:
  `core/lora.py`'s `set_lora_gate()`/`compute_lora_gate()` -- which
  exists specifically to keep the LoRA's contribution close to the
  frozen base outside a dataset's own trained t-range -- had zero wiring
  anywhere in `nodes/` (confirmed via exhaustive grep). A LoRA trained on
  this t range, run through `nodes/`, would apply its full, ungated delta
  at every timestep during generation, including t<150 -- never
  supervised at all. That's a plausible, mechanistic match for "useful
  change and deformation showing up together, worse at higher power" (an
  un-gated delta doesn't get *weaker* at unsupervised timesteps, it gets
  applied at exactly the same strength as everywhere else).

  The gate is now called from `PrepareDiffusionInputsPhase`
  (`nodes/train/step_pipeline.py`) at the same point the legacy pipeline
  calls it. New `gate_enabled` (default `False`, matching the legacy
  default)/`gate_train_low`/`gate_train_high`/`gate_width` ports on
  `SupervisedLoRATrainerNode`. Verified: real `step_pipeline.py` file
  loaded directly (not a copy) with `torch`/`core.lora` mocked to record
  calls -- both the enabled and disabled branches call the right
  functions with the right arguments, and the gate correctly resets to
  `None` when disabled after having been enabled. **Not run** -- needs
  the person to set a restricted `t_low`/`t_high` on a dataset (their own
  earlier report used `[150, 999]` -- a good real case to retest),
  matching `gate_train_low`/`gate_train_high`, and compare LoRA quality
  with `gate_enabled` `True` vs. `False`.

- **[2026-08] VRAM ratchet on non-square datasets -- root cause found,
  fix implemented, not yet confirmed.** Original hypothesis here
  (caching-allocator fragmentation from varying tensor shapes) tested
  directly and found wrong: `num_alloc_retries` stayed `0` across a
  200-step run, `vram_reserved` stayed flat (delta in single-digit MB)
  -- no fragmentation, no growth, on a uniform-shape dataset. On an
  actual non-square dataset, real per-phase VRAM capture (built this
  investigation specifically because a single end-of-step snapshot
  couldn't answer "which phase caused this") caught the real mechanism
  live: the entire VRAM jump (+560MB in one step) happened at exactly
  one phase boundary, `forward`, nowhere else in the step moved at all.
  Root cause: `resize_mode="fit"` preserves aspect ratio with no cap on
  the long side -- a sufficiently tall/wide image forces genuinely
  larger tensors through `forward`, the allocator grabs a bigger
  reserved block and keeps it permanently (nothing in this pipeline
  calls `empty_cache()` on its own). Fix: `manager/builder.py`'s
  `run_lora_ingestion_task` gets a new `max_aspect_ratio` parameter
  (default `2.0`) -- images whose "fit"-resized long side would exceed
  it get split into multiple same-caption crops instead of one
  oversized sample. Wired through the UI
  (`server/static/dataset_manager.html`/`.js`). Crop-box arithmetic
  verified against the actual committed function. **Not run** -- needs
  the person to re-ingest a non-square dataset with the new cap and
  confirm `vram_reserved` stays bounded.

- **[2026-08] CAME `optimizer_step` ~7x AdamW's -- confirmed structural,
  not the old "Chunked vs. Foreach host-sync" hypothesis, fix
  implemented, not yet confirmed.** Original hypothesis here (wrong
  optimizer node in use, "Chunked" instead of "Foreach") tested directly
  and found wrong: `ForeachCAMEOptimizerNode` and
  `ComposedCAMEOptimizerNode` (any strategy) showed the same ~1041ms
  `optimizer_step`, and running the legacy `CAMEOptimizerNode` through
  the current, properly-synchronized profiler gave the same number too
  -- ruling out both "wrong node" and "the old pre-nodes/ pipeline was
  genuinely faster" (its own timer, `core/timer.py`'s `StepTimer`, has
  zero `synchronize()` calls anywhere in it, so it never measured real
  GPU execution time to begin with). Real cause, confirmed by direct
  A/B: CAME vs. AdamW, same everything else, `1041ms` vs. `148ms` -- CAME's
  math runs as an un-batched per-parameter Python loop in every current
  implementation (confirmed by reading all four: `ChunkedXPUCAME`,
  `ForeachXPUCAME`, and every `ComposedCAMEOptimizerNode` strategy).
  Fix: new `ShapeGroupedBatchStrategy`
  (`nodes/optimizer/strategies/shape_grouped.py`), groups parameters by
  exact shape/dtype/device/lr and runs each group's entire update as one
  batched computation. Equivalence-verified via a numpy transcription of
  the exact math (~7e-7 max relative difference across group sizes
  1/2/5/20 and multiple shapes) and a shipped smoke test
  (`nodes/smoke_tests/smoke_test_shape_grouped_equivalence.py`). **Not
  run** -- needs the person to run that smoke test on real hardware,
  then a `profile=True` comparison against the `1041ms` baseline with
  `strategy="shape_grouped"`.

- **[2026-08] Same root cause confirmed for Adafactor too, and now fixed
  for both -- real user report, not just the CAME finding above.** User
  reported real, felt slowness on real hardware: AdamW 3-4x faster than
  Adafactor/CAME, specifically worse on a weak CPU, and "even legacy
  code converted into nodes (adafactor/CAME)" feeling slower. Two things
  checked directly, not assumed:

  1. **The node wrapper itself is not the cause.**
     `AdafactorOptimizerHandle.step()`/`CAMEOptimizerHandle.step()`
     (`nodes/optimizer/adafactor.py`/`came.py`) are one-line pass-throughs
     to the exact same legacy `core.optimizers` classes `core/trainer.py`
     calls directly -- confirmed by reading both. Negligible Python call
     overhead either way; wrapping in a node isn't where a real slowdown
     could come from.
  2. **`AdafactorAlgorithm` had the identical gap `CAMEAlgorithm` had
     before the fix above -- confirmed by reading it, not assumed from
     the CAME finding alone.** No `compute_update_batched()` override at
     all, so `ShapeGroupedBatchStrategy` silently fell back to
     `Algorithm`'s own default (loop + stack -- see algorithms/base.py),
     giving zero real batching benefit for Adafactor even when
     `strategy="shape_grouped"` was already selected. `ForeachApplyStrategy`
     doesn't help either for either optimizer -- confirmed by reading
     it: it only batches the final `decay`/`delta` *apply* step via
     `torch._foreach_*`, not the per-parameter algorithm math itself
     (its own module docstring says so directly). So `strategy="chunked"`
     and `strategy="foreach"` both still pay a real per-parameter Python
     loop for Adafactor/CAME's actual update computation -- on a weak
     CPU, where kernel-dispatch/interpreter overhead dominates over the
     (tiny, per-LoRA-matrix) actual compute time, this is exactly the
     kind of cost that shows up as "the optimizer section takes an
     enormous amount of time," and disproportionately worse than a
     faster CPU would show.

  Fix: `AdafactorAlgorithm.compute_update_batched()`
  (`nodes/optimizer/algorithms/adafactor.py`), same pattern as CAME's
  existing override -- covers the common, already-recommended
  `scale_parameter=False` case. Real, documented scope boundary, not
  papered over: `scale_parameter=True` still falls back to the slow
  per-member path, because `alpha_t` (and therefore `decay`) would
  genuinely vary per group member in that mode, breaking
  `compute_update_batched()`'s shared-decay contract -- extending that
  contract is real, separate work with no urgent need yet, since
  `scale_parameter=True` already has its own documented pathology for
  LoRA's zero-initialized B matrix (see `adafactor.py`'s own module
  docstring) and isn't the recommended setting regardless.
  Equivalence-tested (bit-exact, not just within tolerance, across every
  case checked) against the per-member reference --
  `nodes/smoke_tests/smoke_test_adafactor_shape_grouped_equivalence.py`.
  **Not run on real hardware** -- same caveat as CAME's own entry above:
  needs the person to run `strategy="shape_grouped"` on both Adafactor
  and CAME and compare against their own real `1041ms`-class baseline.

  One more real, honest caveat on the "GPU AdamW is 3-4x faster" part of
  the report: this is *expected*, not itself a bug. `SimpleAdamWOptimizerNode`
  wraps `torch.optim.AdamW(foreach=True)` -- PyTorch's own first-party
  batched multi-tensor kernels, real batching Adafactor/CAME's own math
  has never had until the fix above. Even with `shape_grouped`, some gap
  between AdamW and Adafactor/CAME is genuinely expected (more reduction
  operations per parameter, not just an execution-overhead difference) --
  the open question `shape_grouped` is meant to answer is whether the gap
  shrinks to something like that inherent-complexity difference, not
  whether it disappears entirely.


  Reported as compounding slowly (not just a one-time jump), first appeared
  sometime after an earlier preview-VRAM fix (exact point unknown). Ruled
  out two candidates by reading the code: CAME's own memory pool (only
  entered during `optimizer.step()`, which preview never calls) and
  `PreviewGenerator`'s cached conditioning (set once at construction, never
  mutated). Couldn't reproduce or narrow further without XPU hardware, so
  shipped `TRAIN_VRAM_DEBUG=1` env-var-gated diagnostics instead of guessing
  further: 8 checkpoints (`vram_snapshot()` in `comfy_setup.py`) across
  `preview_sampler.py`'s `generate()` (entry, after denoising, after VAE
  load, after decode loop, after `vae.free()`) and `trainer.py`'s
  `_generate_preview()` (entry/before offload, after offload, after
  `generate()` returns, exit/after reload) plus a baseline every 250
  micro-steps during ordinary training. Zero overhead when the env var is
  unset. Waiting on the user to run this and report which checkpoint's
  reading doesn't drop back down across 2-3 consecutive previews.

- **[2026-07] Corrected an overstated claim about unified-teacher LoRA's VRAM
  benefit.** Originally claimed removing the separate teacher model would
  meaningfully reduce steady-state training VRAM. Wrong -- traced the
  existing code and found `self.teacher` was already being moved to CPU
  (`self.teacher.to("cpu")`) right after cache generation, before the main
  training loop starts, in the *original* code too. So the old code's
  resident-during-training VRAM was already just one model, not two; the
  unified-teacher change's real benefit is reducing peak VRAM during the
  (shorter) cache-generation phase specifically, not steady-state training.
  Matches user's report of no measurable change in their monitored training
  VRAM after applying that patch.
