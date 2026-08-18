"""AdafactorAlgorithm: pure per-parameter Adafactor update math.

Reference: Shazeer & Stern, "Adafactor: Adaptive Learning Rates with
Sublinear Memory Cost" (ICML 2018), as implemented in this codebase's own
core/optimizers.py ChunkedXPUAdafactor. Covers the factored/non-factored
second-moment estimation with Adafactor's time-varying `rho_t` decay
schedule (distinct from CAME's fixed EMA betas -- see begin_step() for
why that needs its own hook), raw-gradient RMS clipping, optional
momentum (`beta1`), `scale_parameter` (on and off), and `weight_decay`.

Does not cover the tiny-parameter batching fast path -- a strategy/
batching concern, not an algorithm one (see algorithms/base.py).
compute_update_batched() batches the common (scale_parameter=False)
case for ShapeGroupedBatchStrategy -- see that method's own docstring
for why scale_parameter=True falls back to the per-member default
instead.

In-place scratch reuse (when `scratch is not None`) needs the shared
buffer reused only once, since Adafactor's formula has one normalization
stage (unlike CAME's two) -- see algorithms/came.py for the CAME case.

**Numerical caveat worth knowing, not a bug:** with `scale_parameter=True`,
the effective step size is `clamp(param_rms**2, min=max(eps1,eps2**2)) *
lr`. For a parameter initialized at or near zero (LoRA's B matrix, by
convention, is initialized to exactly zero), `p_rms` starts near zero, so
the clamp floor dominates and `alpha_t` collapses to roughly
`1e-6 * lr` -- updates stay tiny, the parameter stays near zero, and
`alpha_t` stays near the floor: a self-reinforcing near-standstill. Both
this implementation and the legacy reference reproduce this identically.
`scale_parameter=False` has no such dependency on the parameter's own
magnitude -- effective step size is just `lr`.

See nodes/smoke_tests/smoke_test_adafactor_equivalence.py and
nodes/smoke_tests/smoke_test_composed_adafactor.py for the equivalence
verification against core/optimizers.py's reference implementation.
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

    def compute_update_batched(self, grad_stack, params: list, states: list[dict],
                                lr: float):
        """The same math as _compute_update_safe's factored branch above,
        with one leading group axis (k) threaded through every reduction --
        see algorithms/came.py's own compute_update_batched() for the
        precedent this follows closely (state stacked fresh, computed,
        scattered back via .copy_(); a device-tensor torch.clamp() in
        place of a host-syncing Python min/max).

        **Real scope boundary, not an oversight: only handles
        scale_parameter=False.** With scale_parameter=True, alpha_t
        depends on each parameter's own live norm (`param.data.norm()`)
        -- genuinely different per group member, not a shared
        group-level constant the way it is here. Worse, `decay` is
        derived from alpha_t (`1.0 - wd*alpha_t`), so it would ALSO vary
        per member -- breaking compute_update_batched()'s own contract
        that decay is shared across a whole group (see algorithms/base.py's
        docstring, the same assumption CAMEAlgorithm's batched override
        already relies on). Extending that contract to a per-member decay
        is real, separate work with no urgent need yet: scale_parameter=True
        already has its own documented pathology for LoRA's zero-initialized
        B matrix (see this module's own docstring), and
        composed_adafactor.py's own recommended default is
        scale_parameter=False -- the case this method actually batches is
        also the recommended one. Falls back to the base class's
        per-member default when scale_parameter=True, same shape as
        CAMEAlgorithm's own 1D fallback.

        clip_mul is a genuine per-member vector even in the batched case
        (each member's own gradient norm this step) -- that's fine, it
        never touches decay, only alpha_t does.

        See nodes/smoke_tests/smoke_test_adafactor_shape_grouped_equivalence.py
        for the numerical equivalence check against a per-member
        AdafactorAlgorithm (tolerance-based, not bit-exact -- same
        reduction-order caveat as CAMEAlgorithm's own batched override)."""
        import torch

        if self.scale_parameter:
            return Algorithm.compute_update_batched(self, grad_stack, params, states, lr)
        if self._rho_t is None:
            raise RuntimeError(
                "AdafactorAlgorithm.compute_update_batched() called before begin_step() "
                "-- see compute_update()'s own RuntimeError for why this must run first."
            )
        rho_t = self._rho_t
        k = grad_stack.shape[0]
        n = grad_stack[0].numel()  # same for every member -- exact-shape grouping

        alpha_t = max(self.eps1, 1.0) * lr  # group-uniform -- see docstring above
        decay = (1.0 - self.wd * alpha_t) if self.wd != 0 else None
        factored = grad_stack.dim() >= 3

        flat = grad_stack.reshape(k, -1)
        rms_g = flat.norm(dim=1) / (n ** 0.5 + 1e-8)
        clip_mul = torch.clamp(self.clip_threshold / rms_g, max=1.0)  # per-member, see docstring
        g = grad_stack * clip_mul.view(k, *([1] * (grad_stack.dim() - 1)))

        if not factored:
            g2 = g.pow(2)
            vs_stack = torch.stack([s["vs"] for s in states], dim=0)
            vs_stack.mul_(rho_t).add_(g2.add(self.eps1), alpha=1.0 - rho_t)
            normalized = g / vs_stack.sqrt().add(self.eps1)

            if self.beta1 is not None:
                ea_stack = torch.stack([s["exp_avg"] for s in states], dim=0)
                ea_stack.mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
                normalized = ea_stack.clone()

            delta_stack = normalized * alpha_t

            for j, s in enumerate(states):
                s["vs"].copy_(vs_stack[j])
                if self.beta1 is not None:
                    s["exp_avg"].copy_(ea_stack[j])
            return delta_stack, decay

        rows = grad_stack.shape[1]
        g_view = g.reshape(k, rows, -1)
        g2 = g_view.pow(2)
        vr_stack = torch.stack([s["vr"] for s in states], dim=0)
        vc_stack = torch.stack([s["vc"] for s in states], dim=0)
        vr_stack.mul_(rho_t).add_(g2.mean(dim=2).add(self.eps1), alpha=1.0 - rho_t)
        vc_stack.mul_(rho_t).add_(g2.mean(dim=1).add(self.eps1), alpha=1.0 - rho_t)
        vr_mean_sqrt = vr_stack.mean(dim=1).add(self.eps1).sqrt()
        vr_sqrt = vr_stack.sqrt().add(self.eps1)
        vc_sqrt = vc_stack.sqrt().add(self.eps1)
        normalized = (g_view / vr_sqrt.unsqueeze(2) / vc_sqrt.unsqueeze(1)
                      * vr_mean_sqrt.view(k, 1, 1))
        normalized = normalized.reshape(grad_stack.shape)

        if self.beta1 is not None:
            ea_stack = torch.stack([s["exp_avg"] for s in states], dim=0)
            ea_stack.mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
            normalized = ea_stack.clone()

        delta_stack = normalized * alpha_t

        for j, s in enumerate(states):
            s["vr"].copy_(vr_stack[j])
            s["vc"].copy_(vc_stack[j])
            if self.beta1 is not None:
                s["exp_avg"].copy_(ea_stack[j].reshape(s["exp_avg"].shape))

        return delta_stack, decay
