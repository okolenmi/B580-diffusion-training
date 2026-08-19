"""ComposedAdamWOptimizerNode: AdamWAlgorithm + a selectable ExecutionStrategy.

Same relationship to adamw.py's AdamWOptimizerNode/SimpleAdamWOptimizerNode
as ComposedCAMEOptimizerNode has to CAMEOptimizerNode. Unlike CAME/
Adafactor's composed nodes, this one has no legacy core.optimizers class
it's meant to eventually replace end-to-end for the CPU path: CPUAdamW's
own CPU-resident design is a real, different tradeoff (see adamw.py's
module docstring), not something this Node's device-resident math is
trying to reproduce. What it does replace is *AdamWOptimizerNode's need
to import core.optimizers at all* for anyone who wants a device-resident,
strategy-selectable AdamW built entirely from this package's own
Algorithm/ExecutionStrategy pieces.

Verified against CPUAdamW's own formula directly (same bias-corrected-lr
AdamW variant, same decoupled-decay-at-base-lr convention) --
see nodes/smoke_tests/smoke_test_adamw_equivalence.py.

`strategy="foreach"` is `ForeachApplyStrategy` -- included here (and in
composed_came.py/composed_adafactor.py) because it's algorithm-agnostic
by construction: no AdamW-specific code was needed to add it.
`strategy="shape_grouped"` (`ShapeGroupedBatchStrategy`) is real here
too -- AdamWAlgorithm.compute_update_batched() has no factored reduction
and no clip-based division to worry about (see that method's own
docstring), so it was a small, low-risk addition once CAME's and
Adafactor's own batched overrides had already established the pattern.
The set of valid `strategy` names lives in one place now,
strategy_registry.py -- see that module's docstring for why.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.adamw import AdamWAlgorithm
from .composed import ComposedOptimizerHandle, ParameterGroupPolicy
from .handle import OptimizerHandle
from .node import OptimizerNode
from .strategy_registry import STRATEGY_DOC, resolve_strategy


class ComposedAdamWOptimizerNode(OptimizerNode):
    """AdamW, composed from a pure-math Algorithm + a selectable
    ExecutionStrategy -- device-resident, no core.optimizers import."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999)),
        "eps": Port(name="eps", type=float, required=False, default=1e-8),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=1e-2),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "strategy": Port(name="strategy", type=str, required=False, default="simple",
                          doc=STRATEGY_DOC),
        "group_policy": Port(
            name="group_policy", type=ParameterGroupPolicy, required=False, default=None,
            doc="None = UniformGroups (every parameter at the base lr). "
                "LoRAPlusGroups(...) trains LoRA's B matrices at a higher rate than A -- "
                "see nodes/optimizer/composed.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        algorithm = AdamWAlgorithm(
            betas=inputs.get("betas", self.INPUTS["betas"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
        )
        strategy_name = inputs.get("strategy", self.INPUTS["strategy"].default)
        strategy = resolve_strategy(strategy_name)
        handle = ComposedOptimizerHandle(
            algorithm=algorithm,
            strategy=strategy,
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
            group_policy=inputs.get("group_policy"),
        )
        result = {"optimizer": handle}
        self.validate_outputs(result)
        return result
