*[← docs/known-issues index](README.md)*

# Resolved

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
