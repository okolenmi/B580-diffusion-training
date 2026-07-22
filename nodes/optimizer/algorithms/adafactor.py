"""AdafactorAlgorithm: pure per-parameter Adafactor update math.

This is the second `Algorithm` built in this package, and the answer to
`docs/nodes_package_design.md`'s "one open architectural question": does
the Algorithm/ExecutionStrategy split, only ever proven with CAME so far,
actually generalize to a second, differently-shaped algorithm? Short
answer: mostly yes, with one real, precisely-scoped exception -- see
below.

Reference is core/optimizers.py's ChunkedXPUAdafactor (this codebase's
own already-verified, production implementation of Shazeer & Stern's
"Adafactor: Adaptive Learning Rates with Sublinear Memory Cost", ICML
2018) -- not an external repo, since this codebase's own version is
already the thing any faithful reimplementation needs to match. A fresh,
from-scratch reimplementation (not a wrapper), same discipline as
`algorithms/came.py`.

**What this covers**: the factored/non-factored second-moment estimation
with Adafactor's time-varying `rho_t` decay schedule (distinct from
CAME's fixed EMA betas -- see `begin_step()` below for why that needed a
new hook), raw-gradient RMS clipping, and optional momentum (`beta1`).
Verified numerically against `ChunkedXPUAdafactor` directly (real torch,
not a hand-transcribed reference) -- see verification note at the bottom
of this docstring.

**What this deliberately does NOT cover, precisely stated rather than
silently dropped** (see `algorithms/base.py`'s module docstring for the
fuller architectural reasoning):

- `scale_parameter=True` (the legacy default). The reference's effective
  step size in that mode is `clamp(param_rms**2, min=...) * lr` --
  dependent on both `lr` and the *live parameter's own current
  magnitude*, neither of which `compute_update(grad, state, scratch)`
  has access to. This class only implements the `scale_parameter=False`
  case, where the effective step size reduces to
  `max(eps1, 1.0) * lr`, which for any realistic `eps1 < 1` (the default
  is `1e-8`) is exactly `lr` -- fitting the existing "Algorithm returns
  an unlabeled update, ExecutionStrategy applies `-lr *`" contract
  precisely, no interface change needed. Supporting `scale_parameter=True`
  faithfully would need `compute_update()` to receive `lr` and the live
  parameter tensor -- a real, separate extension, not attempted here.
- Weight decay. The reference applies it as `p *= 1 - wd*alpha_t` --
  a *multiplicative* rescale of the live parameter, directly coupled to
  `alpha_t` (itself `lr`/`scale_parameter`-dependent) -- not an additive
  delta `compute_update()` could express even with `lr` added to its
  signature. Needs its own decision (e.g. a separate
  `Algorithm.weight_decay_scale(lr) -> float | None` hook an
  ExecutionStrategy could apply generically) -- not guessed at here.
- The tiny-parameter batching fast path -- a strategy/batching concern
  by this package's own established reasoning (see `algorithms/base.py`
  and `algorithms/came.py`'s module docstrings), not an algorithm one.
- Using the `scratch` hint for its own internal intermediates (axis 2 of
  the two-axis distinction in `docs/nodes_package_design.md`) --
  deliberately not attempted, same judgment call `CAMEAlgorithm` already
  made: getting in-place buffer reuse subtly wrong was a real, previously
  caught bug in `core/optimizers.py`'s `ChunkedXPUCAME`, and this class
  makes the same conservative choice for the same reason.

**Verification, stated precisely:** compared step-by-step against
`core.optimizers.ChunkedXPUAdafactor` directly (real torch, CPU --
possible now that torch installs in this session's environment, unlike
the numpy-backed mock `CAMEAlgorithm` was originally checked with), with
`scale_parameter=False, weight_decay=0` to match this class's scope.
Parameters sized >= 10,000 elements -- below that, the reference routes
through its tiny-parameter batching fast path (a completely different,
deliberately-unported code path, not what this class implements; an
early version of this check used small toy parameters and appeared to
show large discrepancies for exactly this reason, before being caught
and corrected).

1. **Core formula (no momentum): matches to float32 precision.**
   Factored and non-factored branches, the `rho_t` schedule, and RMS
   clipping -- both `SimpleLoopStrategy` and `ChunkedScratchBufferStrategy`
   -- max abs diff ~2e-6 (relative ~5e-6) over 40 steps against the
   reference. A real, small, bounded first-step cold-start difference
   exists (state here is always pre-allocated to zero by `init_state()`
   -- see `algorithms/base.py`'s docstring for why that's a fixed
   package-wide invariant -- vs. the reference's hard-set-on-first-use;
   at `t=1`, `rho_t=1e-4`, so the blend is `1e-4 * 0 + 0.9999 * new`, a
   ~0.01% difference from the reference's exact value on step 1 only)
   but it's dominated by ordinary float32 rounding noise in this check.

2. **Momentum (`beta1` set): a real, separate, non-algorithmic
   discrepancy was found and fully explained, not shrugged off.** An
   initial comparison using float32 parameters showed a large (~40%
   relative) divergence whenever `beta1` was set. Traced to
   `core/optimizers.py`'s `ChunkedXPUAdafactor.step()` itself:
   `p.data.sub_(g.to(dtype=p.dtype).mul_(alpha_t))`, where `g` is
   `self.exp_avg[i]` (aliased, not copied, a few lines above). When the
   trained parameter's dtype equals the state's float32 dtype exactly,
   `.to(dtype=p.dtype)` is a documented no-op returning the *same tensor
   object* (confirmed directly: `t.to(dtype=t.dtype) is t` -> `True`) --
   so `.mul_(alpha_t)` right after it mutates `self.exp_avg[i]` in place,
   permanently shrinking the momentum buffer by `alpha_t` (~`lr`) every
   step. This is a genuine, previously-undocumented edge case in the
   legacy reference -- but it only manifests when a parameter's own
   dtype is float32; real training here uses bf16 parameters, where
   `.to(dtype=bf16)` from float32 always allocates a fresh tensor, so
   the aliasing (and the corruption) never happens. Re-ran the same
   comparison with bf16 parameters (matching real usage) and the
   divergence collapsed to ~4e-3 absolute (parameters ~0.1 scale) --
   smaller than the `beta1=None` case's own bf16 divergence (~9e-3),
   confirming momentum isn't adding any *extra* error once the aliasing
   artifact is removed; both are just ordinary bf16 quantization noise
   (checked it grows mildly and sub-linearly over 40 steps, not
   exploding, consistent with rounding noise rather than a hidden bug).
   `AdafactorAlgorithm.compute_update()` returns `state["exp_avg"].clone()`
   deliberately, never the aliased tensor itself -- so this class never
   had the corruption to begin with; noted here as a finding about the
   reference, not a defect this port needed to work around. Recorded in
   `docs/suspicious_findings.md` as a new, informational, low-priority
   entry (not fixed -- `nodes/` never edits `core/`, and it doesn't
   affect real bf16 training).

See `nodes/smoke_tests/smoke_test_adafactor_equivalence.py` for the
executable form of this comparison, and
`nodes/smoke_tests/smoke_test_composed_adafactor.py` for the
toy-regression + lifecycle-method check mirroring
`smoke_test_composed_came.py`'s.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import Algorithm


class AdafactorAlgorithm(Algorithm):

    def __init__(self, eps=(1e-8, 1e-3), clip_threshold: float = 1.0,
                 beta1: float | None = None):
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1 = beta1
        self.t = 0
        self._rho_t: float | None = None

    def begin_step(self, n_steps: int = 1) -> None:
        """Advance the shared step counter and compute this step's rho_t
        ONCE -- not per parameter. See algorithms/base.py's begin_step()
        docstring for why this hook exists at all: Adafactor's rho_t is
        genuinely once-per-real-step state, unlike anything CAME needs.
        Matches ChunkedXPUAdafactor's own `self.t += n_steps` (at the top
        of step(), before its per-parameter loop) exactly.
        """
        self.t += n_steps
        self._rho_t = max(1e-4, 1.0 - self.t ** -0.8)

    def init_state(self, param_shape, dtype, device) -> dict[str, Any]:
        """dtype accepted (Algorithm contract) but unused -- state is
        always float32 regardless of the parameter's own dtype, matching
        ChunkedXPUAdafactor's verified behavior (see CAMEAlgorithm's
        init_state() docstring for the same note, which applies
        identically here)."""
        if len(param_shape) >= 2:
            rows = param_shape[0]
            cols = 1
            for d in param_shape[1:]:
                cols *= d
            state: dict[str, Any] = {
                "vr": torch.zeros(rows, dtype=torch.float32, device=device),
                "vc": torch.zeros(cols, dtype=torch.float32, device=device),
            }
        else:
            state = {
                "vs": torch.zeros(param_shape, dtype=torch.float32, device=device),
            }
        if self.beta1 is not None:
            state["exp_avg"] = torch.zeros(param_shape, dtype=torch.float32, device=device)
        return state

    def compute_update(self, grad, state: dict[str, Any], scratch=None):
        """scratch accepted for interface compatibility but not used for
        in-place restructuring -- see module docstring's "what this does
        NOT cover" section for why. grad is treated as read-only: every
        op below produces a new tensor rather than mutating `grad` (or a
        `scratch` view of it) in place, deliberately matching
        CAMEAlgorithm's own conservative choice."""
        if self._rho_t is None:
            raise RuntimeError(
                "AdafactorAlgorithm.compute_update() called before begin_step() -- "
                "every ExecutionStrategy must call algorithm.begin_step(n_steps) "
                "once, before its per-parameter loop, so rho_t reflects the current "
                "real step rather than being unset. See algorithms/base.py's "
                "begin_step() docstring."
            )
        rho_t = self._rho_t
        n = grad.numel()

        # Raw-gradient RMS clipping -- Adafactor's own clip, applied to the
        # raw gradient before any second-moment normalization (distinct
        # from CAME's post-normalization clip in algorithms/came.py).
        rms_g = grad.norm() / (n ** 0.5 + 1e-8)
        clip_mul = min(1.0, self.clip_threshold / float(rms_g))
        g = grad if clip_mul == 1.0 else grad * clip_mul

        factored = g.dim() >= 2
        if factored:
            g_view = g.reshape(g.shape[0], -1)
            g2r = g_view.pow(2).mean(dim=1)
            g2c = g_view.pow(2).mean(dim=0)
            state["vr"].mul_(rho_t).add_(g2r.add(self.eps1), alpha=1.0 - rho_t)
            state["vc"].mul_(rho_t).add_(g2c.add(self.eps1), alpha=1.0 - rho_t)
            vr_mean_sqrt = state["vr"].mean().add(self.eps1).sqrt()
            vr_sqrt = state["vr"].sqrt().add(self.eps1)
            vc_sqrt = state["vc"].sqrt().add(self.eps1)
            normalized = g_view / vr_sqrt.unsqueeze(1) / vc_sqrt.unsqueeze(0) * vr_mean_sqrt
            normalized = normalized.reshape(g.shape)
        else:
            g2 = g.pow(2)
            state["vs"].mul_(rho_t).add_(g2.add(self.eps1), alpha=1.0 - rho_t)
            normalized = g / state["vs"].sqrt().add(self.eps1)

        if self.beta1 is not None:
            state["exp_avg"].mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
            return state["exp_avg"].clone()
        return normalized

    def decay_state(self, state: dict[str, Any], factor: float) -> None:
        if factor <= 0:
            return self.reset_state(state)
        for t in state.values():
            t.mul_(factor)

    def reset_state(self, state: dict[str, Any]) -> None:
        """Zeroes vr/vc/vs/exp_avg only -- matches ChunkedXPUAdafactor's
        own reset_states(), which likewise leaves self.t (and therefore
        the rho_t schedule) untouched. self.t/`_rho_t` live on this
        Algorithm instance, outside the per-parameter `state` dict
        ComposedOptimizerHandle manages, so they're naturally unaffected
        by reset_states()/free_states() either way -- no special-casing
        needed for that parity to hold."""
        for t in state.values():
            t.zero_()
