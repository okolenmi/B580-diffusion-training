"""AdafactorAlgorithm: pure per-parameter Adafactor update math.

This is the second `Algorithm` built in this package, and the answer to
`docs/nodes_package_design.md`'s "one open architectural question": does
the Algorithm/ExecutionStrategy split, only ever proven with CAME so far,
actually generalize to a second, differently-shaped algorithm? Short
answer: yes, once `compute_update()`'s contract was extended to receive
`lr` and the live parameter (see `algorithms/base.py`'s module docstring
for that extension and why it was needed) -- see below.

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
new hook), raw-gradient RMS clipping, optional momentum (`beta1`),
`scale_parameter` (both on and off), and `weight_decay` -- the full
configuration surface of the legacy reference, all verified numerically
against it directly (real torch, not a hand-transcribed reference) --
see verification note at the bottom of this docstring.

**What this deliberately does NOT cover:**
- The tiny-parameter batching fast path -- a strategy/batching concern
  by this package's own established reasoning (see `algorithms/base.py`
  and `algorithms/came.py`'s module docstrings), not an algorithm one.

**Axis 2 (in-place scratch reuse) -- implemented, and simpler than
`CAMEAlgorithm`'s version, not a copy of it.** `CAMEAlgorithm`'s
in-place path mirrors `core/optimizers.py`'s `ChunkedXPUCAME` fix
closely because that formula genuinely has two sequential normalization
stages (row/col, then a confidence term derived from the residual)
needing the buffer reused twice. Adafactor's formula only has one
normalization stage, so its in-place path needs the buffer reused only
once -- no separate "res" intermediate to juggle. Same gating principle
as CAME, not weakened for convenience: the in-place path only runs when
`scratch is not None`, since `SimpleLoopStrategy` never provides one and
its `grad = p.grad.detach().float()` can alias `p.grad` itself for
float32 parameters (confirmed directly -- see the momentum-aliasing
finding below, and `algorithms/came.py`'s module docstring for the
identical mechanism checked again there).

**A real, independent inefficiency found while designing the in-place
path, fixed in *both* code paths, not just the new one:** the original
factored branch computed `g_view.pow(2)` *twice* -- once for the row
reduction, once for the column reduction -- each allocating a full
extra temporary. Confirmed directly that sharing one `g2 = g_view.pow(2)`
between both `.mean(dim=1)`/`.mean(dim=0)` calls gives bit-identical
results (squaring is a deterministic per-element operation; reading the
result twice instead of recomputing it doesn't change any value). Not
specific to the in-place restructuring at all -- a plain, safe
efficiency fix that happened to be found while looking closely enough
at this code to restructure it.

**A second, independent aliasing hazard checked directly, not
assumed away from the CAME case:** the effective-step-size (`alpha_t`)
computation reads `param.data` (for `scale_parameter`) but never mutates
it, and the in-place path's own state mutations (`vr`/`vc`/`vs`/
`exp_avg`) are always separate persistent tensors, never aliased with
`grad`/`scratch` -- so the only thing that needed checking here was the
same `grad`-aliases-`p.grad` hazard already found for CAME, re-verified
for this class's own call sites rather than assumed to transfer
automatically.

**Verification, exact rather than "close enough":** the same
operation-by-operation reasoning as `CAMEAlgorithm`'s -- every in-place
op performs the identical elementary floating-point operation, in the
identical order, as the (now-shared-`g2`) out-of-place formula, so the
two paths are expected, and confirmed, to be bit-exact. See
`nodes/smoke_tests/smoke_test_adafactor_inplace_equivalence.py`.

**A real, precisely-characterized pathology, found while investigating a
real report of `scale_parameter=True` training appearing to make almost
no progress:** its effective step size is `clamp(param_rms**2, min=
max(eps1,eps2**2)) * lr` -- for a parameter initialized at or near zero
(LoRA's B matrix, by convention, is initialized to exactly zero), `p_rms`
starts at (or very near) zero, so the clamp floor dominates: `alpha_t`
collapses to roughly `1e-6 * lr` (confirmed directly: for
`eps=(1e-8, 1e-3), lr=1e-4`, `alpha_t ≈ 9.9999994e-11`, about a millionth
of plain `lr`). Since updates then stay tiny, the parameter stays near
zero, so `alpha_t` stays near the floor -- a self-reinforcing
near-standstill, not a bug: both this implementation and the legacy
reference reproduce it identically (verified directly, same formula,
same floor). `scale_parameter=False` has no such dependency on the
parameter's own magnitude at all -- effective step size is just `lr`.

**Verification, stated precisely:** compared step-by-step against
`core.optimizers.ChunkedXPUAdafactor` directly (real torch, CPU),
covering the full configuration surface -- both `scale_parameter`
settings, `weight_decay` on and off, momentum on and off, factored and
non-factored parameters, float32 and bf16, both `SimpleLoopStrategy` and
`ChunkedScratchBufferStrategy`. Parameters sized >= 10,000 elements --
below that, the reference routes through its tiny-parameter batching
fast path (a completely different, deliberately-unported code path, not
what this class implements; an early version of this check used small
toy parameters and appeared to show large discrepancies for exactly this
reason, before being caught and corrected).

1. **Core formula (no momentum, scale_parameter=False): matches to
   float32 precision.** Factored and non-factored branches, the `rho_t`
   schedule, and RMS clipping -- max abs diff ~2e-6 (relative ~5e-6) over
   40 steps against the reference, both strategies. A real, small,
   bounded first-step cold-start difference exists (state here is always
   pre-allocated to zero by `init_state()` -- see `algorithms/base.py`'s
   docstring for why that's a fixed package-wide invariant -- vs. the
   reference's hard-set-on-first-use; at `t=1`, `rho_t=1e-4`, so the
   blend is `1e-4 * 0 + 0.9999 * new`, a ~0.01% difference from the
   reference's exact value on step 1 only) but it's dominated by
   ordinary float32 rounding noise in this check.

2. **`scale_parameter=True` (non-zero-initialized parameters, the
   well-behaved regime): matches to float32 precision.** ~6e-8 max abs
   diff (float32) over 40 steps, both strategies -- confirms the
   `p_rms`-based `alpha_t` formula itself is faithfully ported, not just
   the case that reduces to plain `lr`. bf16: ~5e-4, consistent with
   ordinary bf16 rounding noise (see point 4).

3. **`weight_decay != 0`: matches to float32 precision.** ~2e-6 max abs
   diff (`scale_parameter=False`) and ~7e-8 (legacy's own default
   combination, `scale_parameter=True, weight_decay=1.0`) over 40 steps.
   The multiplicative decay applied via the generic `decay` return value
   (see `algorithms/base.py`) matches the reference's own
   `p *= 1 - wd*alpha_t` exactly.

4. **Momentum (`beta1` set): a real, separate, non-algorithmic
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
   `AdafactorAlgorithm.compute_update()` returns a fresh tensor built
   from `state["exp_avg"].clone()`, never the aliased tensor itself --
   so this class never had the corruption to begin with; noted here as a
   finding about the reference, not a defect this port needed to work
   around. Recorded in `docs/suspicious_findings.md` as a new,
   informational, low-priority entry (not fixed -- `nodes/` never edits
   `core/`, and it doesn't affect real bf16 training).

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
                 beta1: float | None = None, scale_parameter: bool = False,
                 weight_decay: float = 0.0):
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1 = beta1
        self.scale_parameter = scale_parameter
        self.wd = weight_decay
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

    def compute_update(self, grad, param, state: dict[str, Any], lr: float, scratch=None):
        """param is read-only, used only for scale_parameter's p_rms.
        See module docstring for the in-place (scratch is not None) vs.
        always-safe (scratch is None) split, and why it's gated that way."""
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

        rms_g = grad.norm() / (n ** 0.5 + 1e-8)
        clip_mul = min(1.0, self.clip_threshold / float(rms_g))

        # Effective step size. scale_parameter=True ties this to the live
        # parameter's own current magnitude -- see module docstring for why
        # this needed `param` and `lr` added to the contract, and for a real,
        # documented pathology this mode has with zero-initialized
        # parameters (e.g. LoRA's B matrix). Reads only param.data, never
        # mutates it -- safe regardless of which path below runs.
        if self.scale_parameter:
            p_rms = param.data.norm(dtype=torch.float32) / (n ** 0.5 + 1e-8)
            alpha_t = float(torch.clamp(p_rms.pow(2), min=max(self.eps1, self.eps2 ** 2)) * lr)
        else:
            alpha_t = max(self.eps1, 1.0) * lr

        decay = (1.0 - self.wd * alpha_t) if self.wd != 0 else None
        factored = grad.dim() >= 2

        if scratch is not None:
            update = self._compute_update_inplace(grad, state, rho_t, alpha_t, clip_mul, factored)
        else:
            update = self._compute_update_safe(grad, state, rho_t, alpha_t, clip_mul, factored)
        return update, decay

    def _compute_update_safe(self, grad, state, rho_t, alpha_t, clip_mul, factored):
        """Always-safe path: never mutates `grad` in place. Used whenever
        `scratch is None` -- see module docstring for exactly why that's
        the right gate."""
        g = grad if clip_mul == 1.0 else grad * clip_mul

        if factored:
            g_view = g.reshape(g.shape[0], -1)
            g2 = g_view.pow(2)  # shared between both reductions below -- see
            # module docstring, this is the independent double-allocation
            # fix, not specific to this being the "safe" path.
            state["vr"].mul_(rho_t).add_(g2.mean(dim=1).add(self.eps1), alpha=1.0 - rho_t)
            state["vc"].mul_(rho_t).add_(g2.mean(dim=0).add(self.eps1), alpha=1.0 - rho_t)
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
            normalized = state["exp_avg"].clone()

        return normalized * alpha_t

    def _compute_update_inplace(self, grad, state, rho_t, alpha_t, clip_mul, factored):
        """In-place path: reuses `grad` (== `scratch`, same object for
        every current caller -- see module docstring) as a workspace for
        the normalized gradient, then (if momentum is on) the final
        momentum-blended update -- one buffer reused at most twice, never
        needing a second full-size temp the way CAMEAlgorithm's res/update
        dance does, since this formula has only one normalization stage.
        Only ever called when `scratch is not None`. Expected, and
        verified, to be bit-exact vs. _compute_update_safe().
        """
        g = grad
        if clip_mul != 1.0:
            g.mul_(clip_mul)

        if factored:
            g_view = g.reshape(g.shape[0], -1)
            g2 = g_view.pow(2)  # one unavoidable full-size temp -- g_view's
            # own (clipped-gradient) value is still needed below for the
            # normalization step, so squaring can't be done in place here
            # without destroying it first.
            state["vr"].mul_(rho_t).add_(g2.mean(dim=1).add(self.eps1), alpha=1.0 - rho_t)
            state["vc"].mul_(rho_t).add_(g2.mean(dim=0).add(self.eps1), alpha=1.0 - rho_t)
            vr_mean_sqrt = state["vr"].mean().add(self.eps1).sqrt()
            vr_sqrt = state["vr"].sqrt().add(self.eps1)
            vc_sqrt = state["vc"].sqrt().add(self.eps1)
            g_view.div_(vr_sqrt.unsqueeze(1))
            g_view.div_(vc_sqrt.unsqueeze(0))
            g_view.mul_(vr_mean_sqrt)
            # g_view (== g == grad) now holds `normalized`, in place.

            if self.beta1 is not None:
                state["exp_avg"].mul_(self.beta1).add_(g_view, alpha=1.0 - self.beta1)
                # g_view's old (normalized) value is no longer needed
                # anywhere after the line above -- safe to overwrite with a
                # copy of exp_avg's (separate storage) current value.
                g_view.copy_(state["exp_avg"])
            g_view.mul_(alpha_t)
            return g.reshape(grad.shape)
        else:
            g2 = g.pow(2)
            state["vs"].mul_(rho_t).add_(g2.add(self.eps1), alpha=1.0 - rho_t)
            g.div_(state["vs"].sqrt().add(self.eps1))
            # g now holds `normalized`, in place.

            if self.beta1 is not None:
                state["exp_avg"].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
                g.copy_(state["exp_avg"])
            g.mul_(alpha_t)
            return g

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
