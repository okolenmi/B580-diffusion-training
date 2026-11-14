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


def apply_update(param, delta, decay) -> None:
    """param.data *= decay (if decay is not None); param.data -= delta.
    The one piece of "how do you push an already-computed (delta, decay)
    onto a parameter" logic every non-batched execution path needs --
    SimpleLoopStrategy's per-parameter loop, ShapeGroupedBatchStrategy's
    own per-member apply, and ComposedFusedOptimizerHandle's per-parameter
    backward hook (composed_fused.py) all call this rather than each
    writing the same three lines. ChunkedScratchBufferStrategy doesn't --
    its apply step is genuinely different (in-place scratch reuse), not
    just a style variant of this one. ForeachApplyStrategy/
    ShapeGroupedForeachStrategy don't either -- see apply_updates_batched()
    right below, their own shared batched equivalent."""
    if decay is not None:
        param.data.mul_(decay)
    param.data.sub_(delta.to(dtype=param.dtype))


def apply_updates_batched(entries: list[tuple]) -> None:
    """Same net effect as calling apply_update(param, delta, decay) once
    per (param, delta, decay) in entries, but batched via
    torch._foreach_*, grouped by (device, dtype) -- entries can span
    multiple groups freely, the grouping happens here.

    Extracted from ForeachApplyStrategy's own step() (foreach.py) once
    ShapeGroupedForeachStrategy (shape_grouped_foreach.py) needed the
    identical batched-apply logic too, rather than a second copy of it --
    see foreach.py's own docstring for the real finding this depends on:
    passing each `decay` as a 0-dim float32 tensor, not a plain Python
    float (the ScalarList overload `torch._foreach_mul_` would otherwise
    take), is load-bearing for bf16 correctness, confirmed by direct
    testing (diverged in ~94% of random trials with the ScalarList form,
    matched exactly with this one) -- not a style choice preserved here
    by accident."""
    import torch

    groups: dict[tuple, list] = {}
    for p, delta, decay in entries:
        groups.setdefault((p.device, p.dtype), []).append((p, delta, decay))

    for (_device, dtype), group_entries in groups.items():
        decayed = [(p, decay) for p, _delta, decay in group_entries if decay is not None]
        if decayed:
            torch._foreach_mul_(
                [p.data for p, _decay in decayed],
                [torch.tensor(decay, dtype=torch.float32) for _p, decay in decayed],
            )
        targets = [p.data for p, _delta, _decay in group_entries]
        deltas = [delta.to(dtype=dtype) for _p, delta, _decay in group_entries]
        torch._foreach_sub_(targets, deltas)


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
