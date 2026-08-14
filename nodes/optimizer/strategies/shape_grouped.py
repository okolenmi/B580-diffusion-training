"""ShapeGroupedBatchStrategy: groups parameters by exact (shape, dtype,
device, lr) and runs each group's update as one batched multi-tensor
computation instead of one pass per parameter.

Every other strategy (SimpleLoopStrategy, ChunkedScratchBufferStrategy,
ForeachApplyStrategy) drives Algorithm.compute_update() once per
parameter in a plain Python loop -- ForeachApplyStrategy batches only the
final apply step via torch._foreach_*, not the per-parameter math itself.
This strategy batches the math too, via Algorithm.compute_update_batched()
(see algorithms/base.py).

Grouping key includes `lr`, not just shape/dtype/device: two
same-shaped parameters with genuinely different effective lr (via
ParameterGroupPolicy, e.g. LoRAPlusGroups) must never be batched under
one shared scalar lr, or one of them would silently get the wrong step
size. Grouping is computed once, lazily, on the first step() call --
parameter shapes are stable for a LoRA's whole training run, no reason to
recompute every step.

Exact-shape grouping only, no padding of near-matching shapes --
padding-based batching is a real extra correctness surface (masking,
gradient leakage through pad regions) for a win exact-shape grouping
already gets most of, since real UNet target modules repeat exact
in/out-feature widths constantly (every q/k/v/out projection at a given
attention width, across many blocks).

A group's membership (shape/dtype/device/lr) is fixed at construction,
but which members actually have a gradient on any given step can vary
(gradient accumulation, partial graphs) -- filtered fresh every call,
same as SimpleLoopStrategy's own `if p.grad is None: continue`.
"""

from __future__ import annotations

from typing import Optional

from .base import ExecutionStrategy, apply_update


class ShapeGroupedBatchStrategy(ExecutionStrategy):

    def __init__(self):
        self._groups: Optional[list[list[int]]] = None

    def _build_groups(self, params, param_lr) -> list[list[int]]:
        groups: dict[tuple, list[int]] = {}
        for i, p in enumerate(params):
            key = (tuple(p.shape), p.dtype, p.device, param_lr[i])
            groups.setdefault(key, []).append(i)
        return list(groups.values())

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        import torch

        algorithm.begin_step(n_steps)
        if self._groups is None:
            self._groups = self._build_groups(params, param_lr)

        for idx_group in self._groups:
            # Gradient accumulation / partial-graph edge case: a group's
            # membership (shape/dtype/device/lr) is fixed at construction,
            # but which members actually have a gradient *this specific
            # step* can vary -- filter fresh every call, same as
            # SimpleLoopStrategy's own `if p.grad is None: continue`.
            live = [i for i in idx_group if params[i].grad is not None]
            if not live:
                continue

            if len(live) == 1:
                i = live[0]
                grad = params[i].grad.detach().float()
                delta, decay = algorithm.compute_update(grad, params[i], states[i], param_lr[i])
                apply_update(params[i], delta, decay)
                continue

            grads = [params[i].grad.detach().float() for i in live]
            grad_stack = torch.stack(grads, dim=0)
            group_params = [params[i] for i in live]
            group_states = [states[i] for i in live]
            lr = param_lr[live[0]]  # identical within a group by construction (grouping key includes lr)
            delta_stack, decay = algorithm.compute_update_batched(
                grad_stack, group_params, group_states, lr)
            for j, i in enumerate(live):
                apply_update(params[i], delta_stack[j], decay)

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
