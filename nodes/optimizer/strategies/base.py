"""ExecutionStrategy: how/when an Algorithm's per-parameter math actually
runs, and how per-parameter state is stored/managed.

Zero algorithm-specific math -- a strategy calls into an Algorithm's
compute_update() for each parameter and applies the result; it has no idea
whether that update came from CAME, Adafactor, or anything else. That's
the whole point of the split (see algorithms/base.py's module docstring):
one strategy implementation works with any Algorithm that satisfies the
Algorithm contract.

This first strategy (simple.py) deliberately does none of the memory
optimizations core/optimizers.py's classes use (scratch-buffer reuse via
an XPU MemPool, torch._foreach_* vectorization, backward-hook fusion) --
proving the Algorithm/Strategy split actually composes correctly is the
goal of this first slice; the memory-optimized strategies are real,
valuable, separate follow-up work once the split itself is validated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ExecutionStrategy(ABC):

    @abstractmethod
    def step(self, algorithm, params, states: list[dict], param_lr: list[float],
              n_steps: int = 1) -> None:
        """Apply one real optimizer update to every parameter in `params`,
        using `algorithm` to compute each parameter's update from its
        gradient and the corresponding entry in `states`, then applying
        `param -= param_lr[i] * update` (or however the strategy chooses
        to combine them -- e.g. a vectorized strategy might batch this).
        States are mutated in place by the algorithm as a side effect.
        """

    @abstractmethod
    def zero_grad(self, params) -> None:
        ...

    # Offloading/reloading/decaying/resetting the per-parameter `states`
    # list itself is generic -- it doesn't depend on *how* step() runs, so
    # it lives on the composed handle (composed.py), reused by every
    # strategy, rather than being required here. The three hooks below are
    # for a strategy that holds its OWN extra state beyond the per-parameter
    # dicts (e.g. a future scratch-buffer/MemPool strategy) -- default
    # no-ops, only overridden by a strategy that actually has such state.
    # SimpleLoopStrategy (this slice's only strategy so far) has none.
    def offload_extra(self) -> None:
        pass

    def reload_extra(self, device) -> None:
        pass

    def free_extra(self) -> None:
        pass
