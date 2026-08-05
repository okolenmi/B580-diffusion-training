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

ParameterGroupPolicy fixes a real latent bug, closed here before anything
needs it rather than after it ships silently broken: `param_lr` was
already stored as one entry per parameter, but `update_lr()` (called by
the LR schedule every step) unconditionally overwrote every entry with
the same value --
`self.param_lr = [new_lr] * len(self.params)` -- so anything that had
set a per-parameter ratio at construction would have it erased on the
very next step. UniformGroups is the default and reproduces that same
`[lr] * len(params)` exactly, for every existing caller; LoRAPlusGroups
(Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of Large
Models", arXiv:2402.12354, ICML 2024) is the first real consumer once one
is needed, not wired to anything yet -- `is_b_matrix` is a plain
predicate over a parameter, decoupled from any one LoRA implementation's
own naming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .algorithms.base import Algorithm
from .handle import OptimizerHandle
from .strategies.base import ExecutionStrategy


class ParameterGroupPolicy(ABC):
    """One multiplier per parameter (aligned with `params`' order),
    applied to whatever base rate the LRSchedule produces this step."""

    @abstractmethod
    def group_ratios(self, params) -> list[float]:
        ...


class UniformGroups(ParameterGroupPolicy):
    """Every parameter at the same rate -- today's only actual behavior,
    made explicit and the default rather than hardcoded."""

    def group_ratios(self, params) -> list[float]:
        return [1.0] * len(params)


class LoRAPlusGroups(ParameterGroupPolicy):
    """B matrices (zero-initialized) at `ratio`x the base rate, A matrices
    (random-initialized) and everything else at 1x. An infinite-width
    scaling argument (arXiv:2402.12354) shows training both at the same
    rate is inefficient for large-width models; the paper reports up to
    ~2x finetuning speedup and 1-2% task-performance improvement at
    identical computational cost. `ratio=16.0` is a commonly-used
    starting point in public implementations (e.g. Hugging Face PEFT's
    LoraPlusModel), not independently verified as optimal for SDXL LoRA
    here -- a reasonable default to tune from, not a proven-correct
    constant. lambda is tuned per task in the paper too, not
    theoretically pinned to one value -- the theorem gives an asymptotic
    relationship, not a constant."""

    def __init__(self, is_b_matrix: Callable[[Any], bool], ratio: float = 16.0):
        self._is_b_matrix = is_b_matrix
        self._ratio = ratio

    def group_ratios(self, params) -> list[float]:
        return [self._ratio if self._is_b_matrix(p) else 1.0 for p in params]


class ComposedOptimizerHandle(OptimizerHandle):

    def __init__(self, algorithm: Algorithm, strategy: ExecutionStrategy,
                 params, lr: float, device,
                 group_policy: ParameterGroupPolicy | None = None):
        self.algorithm = algorithm
        self.strategy = strategy
        self.params = list(params)
        self.device = device
        self.states = [
            algorithm.init_state(p.shape, p.dtype, device) for p in self.params
        ]
        self._group_ratios = (group_policy or UniformGroups()).group_ratios(self.params)
        self.update_lr(lr)  # single place param_lr gets computed, see below

    @property
    def lr(self) -> float:
        return self._lr

    def update_lr(self, new_lr: float) -> None:
        self._lr = new_lr
        self.param_lr = [new_lr * r for r in self._group_ratios]

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

    def footprint_bytes(self) -> int:
        """Generic over self.states' real shape (list of per-parameter
        state dicts) -- same reason every other lifecycle method here is
        written once: correct for ComposedFusedOptimizerHandle and any
        future Algorithm/ExecutionStrategy pair for free, by construction,
        without writing this again."""
        return sum(t.numel() * t.element_size()
                   for state in self.states for t in state.values())
