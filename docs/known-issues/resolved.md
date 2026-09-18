*[← docs/known-issues index](README.md)*

# Resolved

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
