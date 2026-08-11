"""ShapeGroupedBatchStrategy: docs/optimizer_execution_redesign_plan.md
Phase 2, the actual speed fix that plan exists for.

Every existing strategy (SimpleLoopStrategy, ChunkedScratchBufferStrategy,
ForeachApplyStrategy) drives Algorithm.compute_update() once per parameter,
in a plain Python loop -- confirmed directly for all three before this plan
was written (ForeachApplyStrategy's own docstring: it batches only the
*final* apply step via torch._foreach_*, everything before that, the real
per-parameter math, still runs one parameter at a time regardless of
strategy). For CAME specifically, measured on real hardware: optimizer_step
was ~7x AdamW's at the same batch/resolution, and switching strategy
(simple vs. chunked) changed nothing (1042ms vs. 1041ms) -- strong evidence
the cost is per-parameter-loop overhead (kernel launch count), not memory
management or host syncs, which every existing strategy already avoids or
controls for.

This strategy groups parameters by exact (shape, dtype, device, lr) --
lr included deliberately, not just shape/dtype/device: ComposedOptimizerHandle
already supports per-parameter lr via ParameterGroupPolicy (composed.py's
LoRAPlusGroups, not wired to anything yet but real and already implemented)
-- two same-shaped parameters with genuinely different effective lr must
never be batched under one shared scalar lr, or one of them would silently
get the wrong step size. Grouping is computed once, lazily, on the first
step() call (parameter shapes are stable for a LoRA's whole training run --
no reason to recompute every step).

Real UNet target modules repeat exact in/out-feature widths constantly
(every q/k/v/out projection at a given attention width, across many
blocks), so exact-shape grouping is expected to collapse "100+ individual
parameters" down to a much smaller number of groups -- see
SupervisedLoRATrainerNode's own startup shape-histogram print
(docs/optimizer_execution_redesign_plan.md Phase 0) for the real number
on any given run, rather than assuming.

Deliberately exact-shape grouping only, no padding of near-matching
shapes -- padding-based batching is a real extra correctness surface
(masking, gradient leakage through pad regions) for a win exact-shape
grouping already gets most of.

Status: equivalence-verified only (numpy-level algebraic check for CAME,
see algorithms/came.py's compute_update_batched docstring, plus this
package's own smoke test), NOT YET real-hardware validated -- matching
this project's own established convention (composed_came.py's docstring)
for exactly this kind of change. Not wired as any node's default strategy
yet -- opt-in via strategy="shape_grouped" until a real run confirms both
correctness (bit-exact torch.equal() against SimpleLoopStrategy) and the
actual measured speedup, not just the reasoned expectation of one.
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
