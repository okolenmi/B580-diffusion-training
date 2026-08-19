"""AdamWAlgorithm: pure per-parameter AdamW update math.

Third Algorithm in this package (see algorithms/base.py). Same formula as
core/optimizers.py's CPUAdamW -- bias-corrected lr folded into the step
size, decoupled weight decay applied at the *base* lr, not the
bias-corrected one (matches CPUAdamW's own documented reasoning: using
the bias-corrected lr for decay would make decay strength drift over
early steps, which isn't the intent). Fresh reimplementation, not a
wrapper -- verified against CPUAdamW directly, see
nodes/smoke_tests/smoke_test_adamw_equivalence.py.

t/bias-correction are tracked once per real step via begin_step(), same
pattern as AdafactorAlgorithm's rho_t -- see algorithms/base.py.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .base import Algorithm


class AdamWAlgorithm(Algorithm):

    def __init__(self, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self._bias_correction = 1.0

    def begin_step(self, n_steps: int = 1) -> None:
        self.t += n_steps
        self._bias_correction = (
            math.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1 ** self.t)
        )

    def init_state(self, param_shape, dtype, device) -> dict[str, Any]:
        # Always float32, regardless of the parameter's own dtype -- matches
        # CPUAdamW's own state precision choice.
        return {
            "m": torch.zeros(param_shape, dtype=torch.float32, device=device),
            "v": torch.zeros(param_shape, dtype=torch.float32, device=device),
        }

    def compute_update(self, grad, param, state: dict[str, Any], lr: float, scratch=None):
        # param accepted per the base.py contract but unused -- AdamW's own
        # math needs only the gradient, not the live parameter value.
        m, v = state["m"], state["v"]
        m.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        v.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
        lr_t = lr * self._bias_correction
        delta = (m / v.sqrt().add(self.eps)).mul_(lr_t)
        decay = (1.0 - lr * self.wd) if self.wd != 0 else None
        return delta, decay

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
        """Same math as compute_update(), with one leading group axis (k)
        threaded through -- see algorithms/came.py's/adafactor.py's own
        compute_update_batched() for the precedent. Simpler than either:
        no factored row/col reduction, no clip-based division by a
        potentially-zero tensor, and decay is already group-uniform by
        construction (lr is shared by the grouping key) -- nothing here
        needed a fallback-for-a-harder-case the way Adafactor's
        scale_parameter=True did."""
        m_stack = torch.stack([s["m"] for s in states], dim=0)
        v_stack = torch.stack([s["v"] for s in states], dim=0)
        m_stack.mul_(self.beta1).add_(grad_stack, alpha=1.0 - self.beta1)
        v_stack.mul_(self.beta2).addcmul_(grad_stack, grad_stack, value=1.0 - self.beta2)
        lr_t = lr * self._bias_correction
        delta_stack = (m_stack / v_stack.sqrt().add(self.eps)).mul_(lr_t)
        decay = (1.0 - lr * self.wd) if self.wd != 0 else None

        for j, s in enumerate(states):
            s["m"].copy_(m_stack[j])
            s["v"].copy_(v_stack[j])
        return delta_stack, decay
