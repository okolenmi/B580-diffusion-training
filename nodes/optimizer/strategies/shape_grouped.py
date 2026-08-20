"""ShapeGroupedBatchStrategy: groups parameters by exact (shape, dtype,
device, lr) and runs each group's update as one batched multi-tensor
computation instead of one pass per parameter.

Every other strategy (SimpleLoopStrategy, ChunkedScratchBufferStrategy,
ForeachApplyStrategy) drives Algorithm.compute_update() once per
parameter in a plain Python loop -- ForeachApplyStrategy batches only the
final apply step via torch._foreach_*, not the per-parameter math itself.
This strategy batches the math too, via Algorithm.compute_update_batched()
(see algorithms/base.py). See shape_grouped_foreach.py for the strategy
that combines this with ForeachApplyStrategy's own batched apply --
both axes at once.

The actual grouping and batched-computation logic lives in
shape_grouping.py's build_shape_groups()/compute_grouped_updates() now --
extracted there once shape_grouped_foreach.py needed the identical logic
too, differing only in how the resulting (param, delta, decay) triples
get applied (this strategy: a per-member loop via base.py's
apply_update(); shape_grouped_foreach.py: base.py's batched
apply_updates_batched()). See shape_grouping.py's own docstring for why
that extraction happened, and docs/training_pipeline_design.md section
11.2 for the axis decomposition this is built around.
"""

from __future__ import annotations

from typing import Optional

from .base import ExecutionStrategy, apply_update
from .shape_grouping import build_shape_groups, compute_grouped_updates


class ShapeGroupedBatchStrategy(ExecutionStrategy):

    def __init__(self):
        self._groups: Optional[list[list[int]]] = None

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        algorithm.begin_step(n_steps)
        if self._groups is None:
            self._groups = build_shape_groups(params, param_lr)
        entries = compute_grouped_updates(algorithm, params, states, param_lr, self._groups)
        for p, delta, decay in entries:
            apply_update(p, delta, decay)

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
