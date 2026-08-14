"""CAMEAlgorithm: pure per-parameter CAME update math.

Reference: Luo et al., "CAME: Confidence-guided Adaptive Memory Efficient
Optimization" (ACL 2023), github.com/yangluo7/CAME. A fresh
reimplementation, not a wrapper around core.optimizers.ChunkedXPUCAME.

Deliberate deviation from the bare reference formula: this implementation
adds `eps1` after each `sqrt()` (e.g. `r_sqrt = state["r"].sqrt().add(eps1)`)
as a denominator-safety term the reference (`1/sqrt(r/r.mean())`, no
epsilon) doesn't have -- a bounded, non-compounding numerical-safety
addition, not a formula change.

Contains no GPU memory management, scratch buffers, or batching logic --
see algorithms/base.py's module docstring for why. Any ExecutionStrategy
drives this class one parameter at a time. Weight decay matches the
reference's own decoupled decay exactly: `p *= 1 - wd*lr`, applied via
the generic `decay` return value rather than anything CAME-specific.

`compute_update()` has two code paths per branch (factored/non-factored):
the original out-of-place formulas (used whenever `scratch is None`), and
an in-place-restructured version (used only when `scratch is not None`)
that reuses the same buffer in sequence for the normalized gradient, then
the confidence term `res`, then the final `update`.

**Why the gate is on `scratch`, not just "is a strategy that provides
one".** `SimpleLoopStrategy` never passes `scratch`: its
`grad = p.grad.detach().float()` returns the *same tensor* as `p.grad`
itself whenever a parameter's own dtype is already `float32` (`.float()`
is a no-op cast that returns `self` when the dtype already matches).
Mutating `grad` in place under `SimpleLoopStrategy` would silently
corrupt `p.grad` for any float32 parameter. `ChunkedScratchBufferStrategy`,
by contrast, always populates its buffer via
`grad_view.copy_(p.grad.detach())` first -- a real copy, never aliased
with `p.grad` -- so `scratch is not None` is a reliable signal that the
passed `grad` is safe to mutate. Every current caller passes
`scratch is grad` (the same object) -- this class relies on that
specifically.

See nodes/smoke_tests/smoke_test_composed_came.py for the equivalence
verification (bit-exact comparison between the two code paths, and
against core.optimizers.ChunkedXPUCAME).
"""

from __future__ import annotations

from typing import Any

import torch

from .base import Algorithm


class CAMEAlgorithm(Algorithm):

    def __init__(self, eps=(1e-30, 1e-16), clip_threshold: float = 1.0,
                 betas=(0.9, 0.999, 0.9999), weight_decay: float = 0.0):
        self.eps1, self.eps2 = eps
        self.clip_threshold = clip_threshold
        self.beta1, self.beta2, self.beta3 = betas
        self.wd = weight_decay

    def init_state(self, param_shape, dtype, device) -> dict[str, Any]:
        """dtype is accepted (part of the Algorithm contract -- a future
        algorithm might legitimately want it) but intentionally unused
        here: state is always kept in float32 for numerical stability,
        regardless of the parameter's own dtype (which may be bf16). This
        matches the original ChunkedXPUCAME's verified behavior -- its
        scratch buffer was always float32 too, with the final update cast
        to the parameter's own dtype only at the point it's actually
        applied (see strategies/simple.py's step(), which does exactly
        that: `update.to(dtype=p.dtype)`)."""
        if len(param_shape) >= 2:
            rows = param_shape[0]
            cols = 1
            for d in param_shape[1:]:
                cols *= d
            return {
                "r": torch.zeros(rows, dtype=torch.float32, device=device),
                "c": torch.zeros(cols, dtype=torch.float32, device=device),
                "ea": torch.zeros(param_shape, dtype=torch.float32, device=device),
                "rr": torch.zeros(rows, dtype=torch.float32, device=device),
                "rc": torch.zeros(cols, dtype=torch.float32, device=device),
            }
        # 1D (non-factored) case -- reference doesn't apply the confidence
        # term here at all (its non-factored branch does
        # `update = exp_avg.clone()`), so no rr/rc state is needed for
        # this shape.
        return {
            "s": torch.zeros(param_shape, dtype=torch.float32, device=device),
            "ea": torch.zeros(param_shape, dtype=torch.float32, device=device),
        }

    def compute_update(self, grad, param, state: dict[str, Any], lr: float, scratch=None):
        # param accepted per the base.py contract but unused -- CAME's own
        # math never needed it, only Adafactor's scale_parameter does.
        decay = (1.0 - self.wd * lr) if self.wd != 0 else None
        factored = grad.dim() >= 2

        if scratch is not None:
            return self._compute_update_inplace(grad, state, lr, decay, factored)
        return self._compute_update_safe(grad, state, lr, decay, factored)

    def _compute_update_safe(self, grad, state, lr, decay, factored):
        """Always-safe path: never mutates `grad` in place. Used whenever
        `scratch is None` -- see module docstring for exactly why that's
        the right gate, not merely a cautious default."""
        g = grad.reshape(grad.shape[0], -1) if factored else grad

        if factored:
            g2 = g.pow(2).add(self.eps1)
            state["r"].mul_(self.beta2).add_(g2.mean(dim=1), alpha=1.0 - self.beta2)
            state["c"].mul_(self.beta2).add_(g2.mean(dim=0), alpha=1.0 - self.beta2)
            r_mean_sqrt = state["r"].mean().add(self.eps1).sqrt()
            r_sqrt = state["r"].sqrt().add(self.eps1)
            c_sqrt = state["c"].sqrt().add(self.eps1)
            normalized = g / r_sqrt.unsqueeze(1) / c_sqrt.unsqueeze(0) * r_mean_sqrt

            rms = normalized.norm() / (normalized.numel() ** 0.5 + 1e-8)
            clip_div = max(float(rms / self.clip_threshold), 1.0)
            if clip_div != 1.0:
                normalized = normalized / clip_div

            ea_flat = state["ea"].reshape(g.shape[0], -1)
            ea_flat.mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)

            res = (normalized - ea_flat).pow(2).add(self.eps2)
            state["rr"].mul_(self.beta3).add_(res.mean(dim=1), alpha=1.0 - self.beta3)
            state["rc"].mul_(self.beta3).add_(res.mean(dim=0), alpha=1.0 - self.beta3)
            rr_mean_sqrt = state["rr"].mean().add(self.eps1).sqrt()
            rr_sqrt = state["rr"].sqrt().add(self.eps1)
            rc_sqrt = state["rc"].sqrt().add(self.eps1)
            update = ea_flat / rr_sqrt.unsqueeze(1) / rc_sqrt.unsqueeze(0) * rr_mean_sqrt
            return update.reshape(grad.shape) * lr, decay
        else:
            g2 = g.pow(2).add(self.eps1)
            state["s"].mul_(self.beta2).add_(g2, alpha=1.0 - self.beta2)
            normalized = g / state["s"].sqrt().add(self.eps1)

            rms = normalized.norm() / (normalized.numel() ** 0.5 + 1e-8)
            clip_div = max(float(rms / self.clip_threshold), 1.0)
            if clip_div != 1.0:
                normalized = normalized / clip_div

            state["ea"].mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)
            # Reference does not apply the confidence term for 1D params --
            # momentum is the update directly.
            return state["ea"].clone() * lr, decay

    def _compute_update_inplace(self, grad, state, lr, decay, factored):
        """In-place path: reuses `grad` (== `scratch`, same object for
        every current caller -- see module docstring) as a workspace for
        the normalized gradient, then `res`, then `update` in sequence,
        mirroring core/optimizers.py's ChunkedXPUCAME.step() exactly.
        Only ever called when `scratch is not None`, which is this
        Algorithm's signal that `grad` is safe to mutate -- never called
        from compute_update() otherwise. Expected, and verified, to be
        bit-exact vs. _compute_update_safe() -- same elementary
        floating-point operations in the same order, just written with
        in-place APIs instead of allocating fresh tensors at each step.
        """
        g = grad.reshape(grad.shape[0], -1) if factored else grad

        if factored:
            g2 = g.pow(2).add(self.eps1)  # one small-lived full-size temp --
            # matches ChunkedXPUCAME's own verified pattern exactly, which
            # keeps this one (short-lived, consumed only by the two
            # .mean() calls below) rather than going further than what's
            # already proven correct and hang-free on real hardware.
            state["r"].mul_(self.beta2).add_(g2.mean(dim=1), alpha=1.0 - self.beta2)
            state["c"].mul_(self.beta2).add_(g2.mean(dim=0), alpha=1.0 - self.beta2)
            r_mean_sqrt = state["r"].mean().add(self.eps1).sqrt()
            r_sqrt = state["r"].sqrt().add(self.eps1)
            c_sqrt = state["c"].sqrt().add(self.eps1)
            g.div_(r_sqrt.unsqueeze(1))
            g.div_(c_sqrt.unsqueeze(0))
            g.mul_(r_mean_sqrt)
            # g now holds `normalized`, in place.

            rms = g.norm() / (g.numel() ** 0.5 + 1e-8)
            clip_div = max(float(rms / self.clip_threshold), 1.0)
            if clip_div != 1.0:
                g.div_(clip_div)
            # g now holds clipped `normalized`.

            ea_flat = state["ea"].reshape(g.shape[0], -1)
            ea_flat.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            # ea updated from g's current value -- g's old (normalized)
            # value is no longer needed anywhere after this line, which is
            # exactly what makes reusing it below safe.

            g.sub_(ea_flat).pow_(2).add_(self.eps2)
            # g now holds `res`, in place -- safe, see above.
            state["rr"].mul_(self.beta3).add_(g.mean(dim=1), alpha=1.0 - self.beta3)
            state["rc"].mul_(self.beta3).add_(g.mean(dim=0), alpha=1.0 - self.beta3)
            rr_mean_sqrt = state["rr"].mean().add(self.eps1).sqrt()
            rr_sqrt = state["rr"].sqrt().add(self.eps1)
            rc_sqrt = state["rc"].sqrt().add(self.eps1)
            # g's current (res) value is no longer needed after the two
            # .mean() calls just above -- safe to overwrite with a copy of
            # ea_flat's (separate storage, real values, not aliased) below.
            g.copy_(ea_flat)
            g.div_(rr_sqrt.unsqueeze(1))
            g.div_(rc_sqrt.unsqueeze(0))
            g.mul_(rr_mean_sqrt)
            # g now holds `update`, in place.
            return (g * lr).reshape(grad.shape), decay
        else:
            g2 = g.pow(2).add(self.eps1)
            state["s"].mul_(self.beta2).add_(g2, alpha=1.0 - self.beta2)
            g.div_(state["s"].sqrt().add(self.eps1))
            # g now holds `normalized`, in place.

            rms = g.norm() / (g.numel() ** 0.5 + 1e-8)
            clip_div = max(float(rms / self.clip_threshold), 1.0)
            if clip_div != 1.0:
                g.div_(clip_div)

            state["ea"].mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            # Reference does not apply the confidence term for 1D params --
            # momentum is the update directly. g's old (normalized) value
            # is no longer needed after the line above.
            g.copy_(state["ea"])
            return g * lr, decay

    def decay_state(self, state: dict[str, Any], factor: float) -> None:
        if factor <= 0:
            return self.reset_state(state)
        for t in state.values():
            t.mul_(factor)

    def reset_state(self, state: dict[str, Any]) -> None:
        for t in state.values():
            t.zero_()

    def compute_update_batched(self, grad_stack, params: list, states: list[dict],
                                lr: float):
        """The same math as _compute_update_safe's factored branch above,
        with one leading group axis (k) threaded through every reduction,
        so a whole group of same-shape parameters costs a handful of
        kernel launches total instead of one full pass per member.

        Two deliberate departures from a literal transcription of
        _compute_update_safe:

        1. Only the factored (2D+) branch is batched. The 1D case falls
           back to the base class's per-member default (LoRA weight
           matrices are 2D; this isn't the shape that matters here).
        2. `clip_div` uses `torch.clamp(..., min=1.0)` (stays a device
           tensor, shape (k,)) instead of `max(float(rms/threshold), 1.0)`
           (forces a host sync). Numerically identical -- dividing by the
           same value whether it's a Python float or a device tensor of
           that value. This also removes the `if clip_div != 1.0:`
           skip-when-unclipped branch: dividing by exactly 1.0 changes
           nothing (`x/1.0 == x` exactly in IEEE float), so always
           dividing is safe and needed anyway once clip_div is a
           per-group-member tensor, not a single Python bool you can
           branch on.

        State handling: `states` keeps its existing flat list-of-dicts
        layout (ComposedOptimizerHandle's offload/reload/decay/reset all
        depend on that generic shape and are otherwise untouched by this
        method). This method stacks fresh from each member's own tensors,
        computes, then scatters results back into those same tensors via
        `.copy_()` -- one extra stack+scatter pass per group per step.

        See nodes/smoke_tests/smoke_test_shape_grouped_equivalence.py for
        the numerical equivalence check against a per-member CAMEAlgorithm
        (a tolerance-based comparison, not bit-exact -- restructuring the
        reduction order is expected to introduce a small, bounded amount
        of floating-point difference)."""
        import torch
        k = grad_stack.shape[0]
        decay = (1.0 - self.wd * lr) if self.wd != 0 else None
        factored = grad_stack.dim() >= 3

        if not factored:
            return Algorithm.compute_update_batched(self, grad_stack, params, states, lr)

        rows = grad_stack.shape[1]
        g = grad_stack.reshape(k, rows, -1)
        g2 = g.pow(2).add(self.eps1)

        r_stack = torch.stack([s["r"] for s in states], dim=0)
        c_stack = torch.stack([s["c"] for s in states], dim=0)
        ea_stack = torch.stack([s["ea"].reshape(rows, -1) for s in states], dim=0)
        rr_stack = torch.stack([s["rr"] for s in states], dim=0)
        rc_stack = torch.stack([s["rc"] for s in states], dim=0)

        r_stack.mul_(self.beta2).add_(g2.mean(dim=2), alpha=1.0 - self.beta2)
        c_stack.mul_(self.beta2).add_(g2.mean(dim=1), alpha=1.0 - self.beta2)
        r_mean_sqrt = r_stack.mean(dim=1).add(self.eps1).sqrt()
        r_sqrt = r_stack.sqrt().add(self.eps1)
        c_sqrt = c_stack.sqrt().add(self.eps1)
        normalized = (g / r_sqrt.unsqueeze(2) / c_sqrt.unsqueeze(1)
                      * r_mean_sqrt.view(k, 1, 1))

        flat = normalized.reshape(k, -1)
        rms = flat.norm(dim=1) / (flat.shape[1] ** 0.5 + 1e-8)
        clip_div = torch.clamp(rms / self.clip_threshold, min=1.0)
        normalized = normalized / clip_div.view(k, 1, 1)

        ea_stack.mul_(self.beta1).add_(normalized, alpha=1.0 - self.beta1)

        res = (normalized - ea_stack).pow(2).add(self.eps2)
        rr_stack.mul_(self.beta3).add_(res.mean(dim=2), alpha=1.0 - self.beta3)
        rc_stack.mul_(self.beta3).add_(res.mean(dim=1), alpha=1.0 - self.beta3)
        rr_mean_sqrt = rr_stack.mean(dim=1).add(self.eps1).sqrt()
        rr_sqrt = rr_stack.sqrt().add(self.eps1)
        rc_sqrt = rc_stack.sqrt().add(self.eps1)
        update = (ea_stack / rr_sqrt.unsqueeze(2) / rc_sqrt.unsqueeze(1)
                  * rr_mean_sqrt.view(k, 1, 1))
        delta_stack = (update * lr).reshape(grad_stack.shape)

        for j, s in enumerate(states):
            s["r"].copy_(r_stack[j])
            s["c"].copy_(c_stack[j])
            s["ea"].copy_(ea_stack[j].reshape(s["ea"].shape))
            s["rr"].copy_(rr_stack[j])
            s["rc"].copy_(rc_stack[j])

        return delta_stack, decay
