"""ShapeGroupedForeachStrategy: combines both real batching axes --
ShapeGroupedBatchStrategy's batched core-algorithm math
(Algorithm.compute_update_batched(), grouped by exact shape) and
ForeachApplyStrategy's batched apply step (torch._foreach_*, grouped by
device/dtype). See docs/training_pipeline_design.md section 11.2 for why
these were previously two separate, seemingly non-combinable strategies
(a real question raised directly this session) and what the actual axes
turned out to be once every strategy's step() was read directly rather
than guessed at.

Built entirely from two already-real, already-equivalence-tested pieces,
not re-derived: shape_grouping.py's build_shape_groups()/
compute_grouped_updates() (the same grouping/computation logic
ShapeGroupedBatchStrategy uses) and base.py's apply_updates_batched()
(the same batched-apply logic, including the bf16 ScalarList-vs-tensor-
scalar rounding fix, ForeachApplyStrategy uses). This class's own body is
just wiring the two together -- see nodes/smoke_tests/
smoke_test_shape_grouped_foreach_equivalence.py for the numerical
equivalence check against SimpleLoopStrategy.

Same real, honest scope boundary ShapeGroupedBatchStrategy has:
AdafactorAlgorithm's compute_update_batched() only batches the
scale_parameter=False case (falls back to the per-member default
otherwise -- see that method's own docstring), so this strategy inherits
that boundary for Adafactor specifically, not something new to this
strategy.
"""

from __future__ import annotations

from typing import Optional

from .base import ExecutionStrategy, apply_updates_batched
from .shape_grouping import build_shape_groups, compute_grouped_updates


class ShapeGroupedForeachStrategy(ExecutionStrategy):

    def __init__(self):
        self._groups: Optional[list[list[int]]] = None

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        algorithm.begin_step(n_steps)
        if self._groups is None:
            self._groups = build_shape_groups(params, param_lr)
        entries = compute_grouped_updates(algorithm, params, states, param_lr, self._groups)
        apply_updates_batched(entries)

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
