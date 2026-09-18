"""ComposedOptimizerHandle: generic OptimizerHandle for any Algorithm +
any ExecutionStrategy pair.

This is the actual payoff of the Algorithm/ExecutionStrategy split: the
lifecycle methods (offload_states_to_cpu, reload_states_to_device,
decay_states, reset_states, free_states) are written exactly ONCE, here,
generically over "a list of per-parameter state dicts". Any future
Algorithm or ExecutionStrategy gets these for free by construction.

ParameterGroupPolicy gives one multiplier per parameter, applied on top
of whatever base rate the LRSchedule produces this step. UniformGroups
is the default (every parameter at the same rate). LoRAPlusGroups
(Hayou, Ghosh, Yu, "LoRA+: Efficient Low Rank Adaptation of Large
Models", arXiv:2402.12354, ICML 2024) trains B matrices (zero-initialized)
at a higher rate than A -- `is_b_matrix` is a plain predicate over a
parameter, decoupled from any one LoRA implementation's own naming.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from .algorithms.base import Algorithm
from .handle import OptimizerHandle
from .state_store import Float32StateStore, OptimizerStateStore
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
                 group_policy: ParameterGroupPolicy | None = None,
                 state_store: OptimizerStateStore | None = None):
        self.algorithm = algorithm
        self.strategy = strategy
        self.params = list(params)
        self.device = device
        # Float32StateStore() -- the identity, no compression -- unless a real one is
        # given: see nodes/optimizer/state_store.py's own docstring for the full
        # reasoning (Int8BlockStateStore etc.). self.states below always holds
        # whatever this store's own wrap() returns, never raw Algorithm.init_state()
        # dicts directly -- checkout()/commit() in step()/decay_states()/
        # reset_states() below are what get real, mutable fp32 tensors to
        # Algorithm/ExecutionStrategy, which never see or need to know this exists.
        self._state_store = state_store or Float32StateStore()
        self.states = [
            self._state_store.wrap(algorithm.init_state(p.shape, p.dtype, device))
            for p in self.params
        ]
        self._group_ratios = (group_policy or UniformGroups()).group_ratios(self.params)
        self.update_lr(lr)  # single place param_lr gets computed, see below
        self._offloaded = False

    @property
    def lr(self) -> float:
        return self._lr

    def update_lr(self, new_lr: float) -> None:
        self._lr = new_lr
        self.param_lr = [new_lr * r for r in self._group_ratios]

    def step(self, n_steps: int = 1) -> None:
        # Checkout every state to real fp32 tensors, hand those (not self.states'
        # own handles) to the strategy -- Algorithm.compute_update() never sees
        # self._state_store's own representation, by construction -- then commit
        # whatever compute_update() mutated in place back into self.states.
        checked_out = [self._state_store.checkout(s) for s in self.states]
        self.strategy.step(self.algorithm, self.params, checked_out, self.param_lr, n_steps)
        for handle, state in zip(self.states, checked_out):
            self._state_store.commit(handle, state)

    def zero_grad(self) -> None:
        self.strategy.zero_grad(self.params)

    def offload_states_to_cpu(self) -> None:
        self.states = [self._state_store.to(s, "cpu") for s in self.states]
        self.strategy.offload_extra()
        self._offloaded = True

    def reload_states_to_device(self, device: str | None = None) -> None:
        dev = device if device is not None else self.device
        self.states = [self._state_store.to(s, dev) for s in self.states]
        self.strategy.reload_extra(dev)
        self._offloaded = False

    def decay_states(self, factor: float) -> None:
        # Infrequent (not per-step), so a checkout/commit round trip here -- rather
        # than a separate decay-aware store method -- costs nothing worth avoiding.
        for handle in self.states:
            state = self._state_store.checkout(handle)
            self.algorithm.decay_state(state, factor)
            self._state_store.commit(handle, state)

    def reset_states(self) -> None:
        for handle in self.states:
            state = self._state_store.checkout(handle)
            self.algorithm.reset_state(state)
            self._state_store.commit(handle, state)

    def free_states(self) -> None:
        self.states = []
        self.strategy.free_extra()
        import gc
        gc.collect()

    def footprint_bytes(self) -> int:
        """Generic over self.states' real shape (list of per-parameter
        state handles) -- same reason every other lifecycle method here is
        written once: correct for ComposedFusedOptimizerHandle and any
        future Algorithm/ExecutionStrategy/OptimizerStateStore combination
        for free, by construction, without writing this again.

        0 while offloaded: offload_states_to_cpu() moves the real tensors
        to "cpu" rather than dropping them, so self._state_store's own
        footprint_bytes() would otherwise keep reporting the same
        numel()*element_size() total it always did -- device-memory
        usage is what this promises (see DeviceResident.footprint_bytes()'s
        own docstring), and that's 0 once nothing's actually on a
        device."""
        if self._offloaded:
            return 0
        return sum(self._state_store.footprint_bytes(handle) for handle in self.states)
