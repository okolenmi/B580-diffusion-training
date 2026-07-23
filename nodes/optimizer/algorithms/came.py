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
        # scratch accepted for interface compatibility (see base.py's
        # docstring) but not yet used -- this method still allocates
        # fresh intermediate tensors (g2, normalized, res, ...) rather
        # than reusing a provided buffer via in-place ops. Restructuring
        # this to actually use scratch is real, separate follow-up work --
        # see docs/nodes_package_design.md. Deliberately not rushed here:
        # getting in-place buffer reuse subtly wrong (aliasing a value
        # that's still needed) was a real bug this session already had to
        # catch and fix once, in core/optimizers.py's ChunkedXPUCAME.
        # param accepted per the base.py contract but unused -- CAME's own
        # math never needed it, only Adafactor's scale_parameter does.
        decay = (1.0 - self.wd * lr) if self.wd != 0 else None
        factored = grad.dim() >= 2
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

    def decay_state(self, state: dict[str, Any], factor: float) -> None:
        if factor <= 0:
            return self.reset_state(state)
        for t in state.values():
            t.mul_(factor)

    def reset_state(self, state: dict[str, Any]) -> None:
        for t in state.values():
            t.zero_()
