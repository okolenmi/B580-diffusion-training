"""SimpleLoopStrategy: the simplest possible ExecutionStrategy.

A plain Python for-loop over parameters, calling algorithm.compute_update()
for each and applying the result directly -- no scratch-buffer reuse, no
MemPool, no torch._foreach_* vectorization, no backward-hook fusion. This
is deliberately the least sophisticated strategy possible: the point of
this first slice is proving the Algorithm/ExecutionStrategy split actually
composes into a working optimizer, not matching core/optimizers.py's
memory-optimized classes' performance yet. A ChunkedScratchBufferStrategy
(reusing this session's already-verified scratch-buffer/MemPool pattern)
is real, valuable follow-up work -- but it can be built and tested entirely
independently of any Algorithm, and any Algorithm that satisfies the
Algorithm contract (this session's CAMEAlgorithm, or a future
AdafactorAlgorithm/AdamWAlgorithm) will work with it unchanged the moment
it exists, which is the whole payoff of doing this split properly.
"""

from __future__ import annotations

from .base import ExecutionStrategy


class SimpleLoopStrategy(ExecutionStrategy):

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        algorithm.begin_step(n_steps)
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            update = algorithm.compute_update(grad, states[i])
            p.data.add_(update.to(dtype=p.dtype), alpha=-param_lr[i])

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
