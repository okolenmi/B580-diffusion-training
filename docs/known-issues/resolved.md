*[← docs/known-issues index](README.md)*

# Resolved

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
