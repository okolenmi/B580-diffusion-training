"""ComposedOptimizerHandle: generic OptimizerHandle for any Algorithm +
any ExecutionStrategy pair.

This is the actual payoff of the Algorithm/ExecutionStrategy split: the
lifecycle methods (offload_states_to_cpu, reload_states_to_device,
decay_states, reset_states, free_states) are written exactly ONCE, here,
generically over "a list of per-parameter state dicts" -- rather than
hand-duplicated (with real, found-by-testing inconsistencies -- see
docs/nodes_package_design.md's "course correction" section) across
core/optimizers.py's 5 classes. Any future Algorithm or ExecutionStrategy
gets these for free by construction, correctly, without writing them again.
"""

from __future__ import annotations

from typing import Any

from .algorithms.base import Algorithm
from .handle import OptimizerHandle
from .strategies.base import ExecutionStrategy


class ComposedOptimizerHandle(OptimizerHandle):

    def __init__(self, algorithm: Algorithm, strategy: ExecutionStrategy,
                 params, lr: float, device):
        self.algorithm = algorithm
        self.strategy = strategy
        self.params = list(params)
        self._lr = lr
        self.param_lr = [lr] * len(self.params)
        self.device = device
        self.states = [
            algorithm.init_state(p.shape, p.dtype, device) for p in self.params
        ]

    @property
    def lr(self) -> float:
        return self._lr

    def update_lr(self, new_lr: float) -> None:
        self._lr = new_lr
        self.param_lr = [new_lr] * len(self.params)

    def step(self, n_steps: int = 1) -> None:
        self.strategy.step(self.algorithm, self.params, self.states, self.param_lr, n_steps)

    def zero_grad(self) -> None:
        self.strategy.zero_grad(self.params)

    def offload_states_to_cpu(self) -> None:
        for state in self.states:
            for name, t in state.items():
                state[name] = t.to("cpu", non_blocking=False)
        self.strategy.offload_extra()

    def reload_states_to_device(self, device: str | None = None) -> None:
        dev = device if device is not None else self.device
        for state in self.states:
            for name, t in state.items():
                state[name] = t.to(dev, non_blocking=False)
        self.strategy.reload_extra(dev)

    def decay_states(self, factor: float) -> None:
        for state in self.states:
            self.algorithm.decay_state(state, factor)

    def reset_states(self) -> None:
        for state in self.states:
            self.algorithm.reset_state(state)

    def free_states(self) -> None:
        self.states = []
        self.strategy.free_extra()
        import gc
        gc.collect()
