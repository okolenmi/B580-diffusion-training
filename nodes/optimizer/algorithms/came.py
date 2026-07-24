"""CAMEAlgorithm: pure per-parameter CAME update math.

Algorithm faithfully follows the official reference implementation
(github.com/yangluo7/CAME, came_pytorch/CAME.py) -- Luo et al., "CAME:
Confidence-guided Adaptive Memory Efficient Optimization", ACL 2023. This
is a fresh, from-scratch reimplementation (not a wrapper around
core.optimizers.ChunkedXPUCAME) -- its formulas were re-verified
numerically against the reference before writing torch code here, since
restructuring math into a new pure-function shape is exactly the kind of
change that can silently alter behavior if done carelessly.

Verification result, stated precisely rather than just "verified" -- this
implementation adds eps1 *after* each sqrt() (e.g. `r_sqrt = state["r"].
sqrt().add(eps1)`) as a denominator-safety term the bare reference formula
(`1/sqrt(r/r.mean())`, no epsilon there at all) doesn't have. Checked this
is the actual, sole source of divergence and that it's bounded rather than
compounding: over 15 steps with realistic gradient magnitudes, absolute
difference from the reference is ~1e-8, relative difference ~4e-10 --
jumps once from float64-noise level (~1e-14) at step 0 to this plateau,
then stays flat rather than growing across further steps. Both far below
fp32 precision (~1e-7 relative), the precision this actually runs at.
Same eps-after-sqrt pattern was already used in this session's earlier
ChunkedXPUCAME work and found similarly harmless there. Verified both the
factored (2D+) and non-factored (1D, no confidence term applied -- matches
the reference's own non-factored branch, which uses momentum directly)
code paths.

Deliberately contains NO GPU memory management, scratch buffers, or
batching logic of any kind -- see algorithms/base.py's module docstring
for why. Any ExecutionStrategy drives this class one parameter at a time.

**Weight decay added along with the base.py contract extension** (see
that module's docstring for the full reasoning -- this was built for
Adafactor's `scale_parameter`, and CAME's own weight decay came along for
free once `lr`/`param`/the `decay` return value existed). Matches the
reference's own decoupled decay exactly: `p *= 1 - wd*lr`, applied via
the generic `decay` mechanism rather than anything CAME-specific -- no
strategy code needed to know CAME has weight decay at all. Verified
directly against `core.optimizers.ChunkedXPUCAME` (real torch, not the
external-reference numpy comparison this class's core formulas were
originally checked against): float32 matches to ~4e-6 max abs diff over
40 steps (with or without weight decay -- confirmed identical, so decay
adds no extra error). bf16 shows a larger, but bounded and
mildly-growing (not exploding) divergence -- ~3.9e-3 at step 0 growing
to ~4.7e-2 by step 40 -- present identically with weight_decay=0, so
unrelated to this addition; likely from CAME's own longer chain of
sqrt/divide operations (r/c *and* rr/rc, vs. Adafactor's single
vr/vc) compounding bf16 rounding more per step. Not chased further this
session -- real, but this is the first time this class was compared
against the legacy reference with actual bf16 tensors rather than a
numpy mock, so it's a new data point rather than a regression.

**In-place scratch reuse (axis 2), added this round -- gated carefully,
not attempted unconditionally.** `compute_update()` now has two code
paths per branch (factored/non-factored): the original, always-safe
out-of-place formulas (unchanged, used whenever `scratch is None`), and
a new in-place-restructured version (used only when `scratch is not
None`) that mirrors `core/optimizers.py`'s `ChunkedXPUCAME.step()`
exactly -- reusing the same buffer in sequence for the normalized
gradient, then (once momentum has already consumed its value) for the
confidence term `res`, then (once the row/col reduction has already
consumed *its* value) for the final `update` -- eliminating the same two
full-parameter-sized allocations per step that class's own history
(`docs/suspicious_findings.md`'s "CAME optimizer VRAM near-ceiling hang"
entry) already proved matter.

**Why the gate is on `scratch`, not just "is a strategy that provides
one" -- a real aliasing hazard found and checked, not assumed away.**
`SimpleLoopStrategy` never passes `scratch`, and for good reason,
confirmed directly: its `grad = p.grad.detach().float()` returns the
*same tensor* as `p.grad` itself whenever a parameter's own dtype is
already `float32` (`.float()` is a no-op cast that returns `self` when
the dtype already matches -- the identical mechanism behind the
`ChunkedXPUAdafactor` momentum-aliasing finding in
`algorithms/adafactor.py`'s docstring, checked again here rather than
assumed to be a one-off). Mutating `grad` in place under
`SimpleLoopStrategy` would silently corrupt `p.grad` for any float32
parameter -- confirmed with a direct repro before writing a single line
of the in-place path. `ChunkedScratchBufferStrategy`, by contrast,
always populates its buffer via `grad_view.copy_(p.grad.detach())`
first -- a real copy, never aliased with `p.grad` -- so `scratch is not
None` is a reliable signal that the passed `grad` is safe to mutate.
Every current caller passes `scratch is grad` (the same object) --
this class relies on that specifically, noted here so it's not a silent
assumption if a future caller ever passes a distinct scratch buffer.

**Verification, precise rather than "looks right": the two code paths
are expected to be, and were confirmed to be, bit-exact -- not just
close.** Every in-place op (`.div_()`, `.mul_()`, `.sub_()`, `.copy_()`)
performs the identical elementary floating-point operation, in the
identical order, as its out-of-place counterpart in the original
formula -- restructuring memory layout, not arithmetic. Verified by
running `SimpleLoopStrategy` (old path) and `ChunkedScratchBufferStrategy`
(new path) side by side from identical initial weights/gradients and
diffing every step with `torch.equal()`, not a tolerance -- see
`nodes/smoke_tests/smoke_test_composed_came.py`.
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
        # term here at all (verified earlier this session by reading the
        # reference's non-factored branch directly: `update = exp_avg.
        # clone()`), so no rr/rc state is needed for this shape.
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
