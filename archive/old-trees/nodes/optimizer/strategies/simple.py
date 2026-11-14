"""SimpleLoopStrategy: the simplest possible ExecutionStrategy.

A plain Python for-loop over parameters, calling algorithm.compute_update()
for each and applying the result directly -- no scratch-buffer reuse, no
MemPool, no torch._foreach_* vectorization, no backward-hook fusion. The
least sophisticated strategy, and the baseline every other strategy's
equivalence tests compare against.
"""

from __future__ import annotations

from .base import ExecutionStrategy, apply_update


class SimpleLoopStrategy(ExecutionStrategy):

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        algorithm.begin_step(n_steps)
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            delta, decay = algorithm.compute_update(grad, p, states[i], param_lr[i])
            apply_update(p, delta, decay)

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
