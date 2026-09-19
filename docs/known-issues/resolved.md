*[← docs/known-issues index](README.md)*

# Resolved

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
