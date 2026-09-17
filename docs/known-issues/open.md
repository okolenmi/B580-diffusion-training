*[← docs/known-issues index](README.md)*

# Open

- **[2026-09] `core.optimizers.ChunkedXPUAdafactor`/`FusedXPUAdafactor`
  silently corrupt their own momentum buffer for float32 parameters
  with `beta1` (momentum) set.** `g = self.exp_avg[i]` aliases the
  momentum buffer (no copy); the following
  `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))` calls
  `.to(dtype=p.dtype)`, which for a float32 parameter (state is already
  float32) returns the *same object*, not a copy -- so the subsequent
  `.mul_(alpha_t)` mutates the momentum buffer in place. Net effect:
  every step, right after using the momentum buffer to compute that
  step's update, the buffer gets permanently shrunk by `alpha_t` (~lr)
  as an unintended side effect. Confirmed directly, not theorized --
  see `nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py`'s
  `check_legacy_float32_momentum_bug()` for the `FusedXPUAdafactor`
  case; `ChunkedXPUAdafactor`'s copy at `core/optimizers.py:322` is the
  identical pattern, confirmed by reading it directly, not yet given
  its own smoke-test regression check. bf16 parameters don't trigger
  this (`.to(dtype=p.dtype)` performs a real cast there, producing a
  genuine copy).
  **Scope differs between the two, checked precisely rather than
  assumed:** `FusedXPUAdafactor`'s buggy line is shared by both its
  tiny-parameter (< 10,000 element) and main-parameter code paths, so
  it affects every parameter size. `ChunkedXPUAdafactor`'s buggy line
  sits only in its main-parameter path (`core/optimizers.py:170-322`,
  guarded by `p.numel() < 10_000` routing tiny parameters elsewhere
  instead) -- its own tiny-parameter path applies updates via a
  different mechanism (`ws[s:e].add_(..., alpha=-alpha_t)`, no aliasing
  problem) and is unaffected. `ForeachXPUAdafactor` does not have this
  bug at all -- both its factored and unfactored paths use non-in-place
  `.mul(alpha_t)` (`core/optimizers.py:1045`, `:1105`), which always
  returns a new tensor regardless of dtype, never aliasing the momentum
  buffer. `nodes/optimizer/algorithms/adafactor.py`'s `AdafactorAlgorithm`
  does not have this bug either way -- so `ComposedFusedAdafactorOptimizerNode`
  doesn't inherit it from the legacy class it was verified against, and
  `AdafactorOptimizerNode` (`nodes/optimizer/adafactor.py`) is the only
  node this actually affects now, for float32 parameters with momentum,
  and only for parameters >= 10,000 elements (its own tiny-parameter
  path uses a different, unaffected mechanism). The legacy
  `FusedAdafactorOptimizerNode`/`ForeachAdafactorOptimizerNode` this
  used to also apply to have both since been deleted --
  `ForeachXPUAdafactor` was never affected
  either way, `FusedXPUAdafactor`'s node was retired once
  `ComposedFusedAdafactorOptimizerNode` (bug-free) was confirmed fully
  equivalent otherwise.
  `core/` is untouched by the `nodes/` rewrite (see `docs/architecture.md`),
  so this doesn't get fixed at the source, but it's real and live for
  anyone using `AdafactorOptimizerNode` directly.

  Separately, and not the same issue: `ChunkedXPUAdafactor`/
  `FusedXPUAdafactor` both had a real small-parameter (< 10,000 element)
  fast path -- a plain elementwise second-moment EMA in place of the
  row/col factored approximation -- that `AdafactorAlgorithm` didn't
  cover, and the two didn't even agree with each other on the mechanism
  (`FusedXPUAdafactor`'s is genuinely per-parameter; `ChunkedXPUAdafactor`'s
  ties every tiny parameter in the whole optimizer together into one
  shared clip and EMA state, a cross-parameter batching concern, not a
  per-parameter algorithm branch). `ForeachXPUAdafactor` had no
  tiny-parameter special case at all -- confirmed, not just theorized,
  by an actual torch run:
  `ForeachAdafactorOptimizerNode` came back equivalent to
  `ComposedAdafactorOptimizerNode(strategy="foreach")` at floating-
  point-noise magnitude and has been deleted. `FusedXPUAdafactor`'s gap
  was confirmed real (not noise) the same way, and `AdafactorAlgorithm`
  grew an opt-in fix for it (`tiny_parameter_threshold`, used by
  `ComposedFusedAdafactorOptimizerNode` only) -- confirmed closed on
  real torch (within this project's own already-established equivalence
  tolerances for this exact pair -- `1e-4` float32, `1e-2` bf16, same
  as the main-path case, float32+momentum excluded for the bug above,
  same as the main-path case already excludes it), and
  `FusedAdafactorOptimizerNode` has been deleted too. Permanent
  regression coverage for both fixes lives in
  `nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py`
  (tiny-parameter case) and
  `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py` Part A
  (Foreach case). `ChunkedXPUAdafactor`'s cross-parameter-batching
  version is unrelated to either fix and still fully open -- real,
  separate `ExecutionStrategy`-level feature work, see
  `docs/design/09-prioritized-backlog.md`. `AdafactorOptimizerNode` is
  now the only legacy Adafactor node left registered, kept specifically
  for that gap.

- **[2026-08] `DeviceResident.footprint_bytes()` doesn't check actual
  device placement anywhere.**

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
