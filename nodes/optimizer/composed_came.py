"""ComposedCAMEOptimizerNode: CAMEAlgorithm + a selectable ExecutionStrategy.

A separate class from CAMEOptimizerNode (came.py, which wraps the legacy
core.optimizers.ChunkedXPUCAME) rather than a replacement for it.

The `strategy` input is where the Algorithm/ExecutionStrategy split
becomes usable: switching between strategies changes nothing about
CAMEAlgorithm itself, only how ExecutionStrategy iterates parameters and
manages temporary memory, so choosing between them is purely a memory/
performance decision, never a correctness one. See each strategy's own
module docstring (strategies/simple.py, chunked.py, foreach.py,
shape_grouped.py) and nodes/smoke_tests/ for what each one actually
optimizes and its current equivalence/hardware-validation status. The
set of valid `strategy` names lives in one place now,
strategy_registry.py -- see that module's docstring for why.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.came import CAMEAlgorithm
from .composed import ComposedOptimizerHandle, ParameterGroupPolicy
from .handle import OptimizerHandle
from .node import OptimizerNode
from .strategy_registry import STRATEGY_DOC, resolve_strategy


class ComposedCAMEOptimizerNode(OptimizerNode):
    """CAME, composed from a pure-math Algorithm + a selectable
    ExecutionStrategy."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "lr": Port(name="lr", type=float, required=False, default=1e-4),
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-30, 1e-16)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999, 0.9999)),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0,
                              doc="Decoupled weight decay -- p *= 1 - wd*lr, matching the "
                                  "legacy CAMEOptimizerNode's own default and formula "
                                  "exactly. Added along with algorithms/base.py's lr/param "
                                  "contract extension (built for AdafactorAlgorithm's "
                                  "scale_parameter -- CAME's own weight decay came from the "
                                  "same generic mechanism, not a CAME-specific addition)."),
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
        algorithm = CAMEAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
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
