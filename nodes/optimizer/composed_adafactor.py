"""ComposedAdafactorOptimizerNode: AdafactorAlgorithm + a selectable
ExecutionStrategy.

Same relationship to AdafactorOptimizerNode (adafactor.py, which wraps
the legacy core.optimizers.ChunkedXPUAdafactor) as ComposedCAMEOptimizerNode
has to CAMEOptimizerNode.

`INPUTS` below default to the conservative, predictable values
(`scale_parameter=False, weight_decay=0.0`) rather than
`AdafactorOptimizerNode`'s own legacy defaults (`scale_parameter=True,
weight_decay=1.0`): those legacy defaults are unusual (full weight decay
of 1.0 shrinks any parameter by ~5% per step at a typical lr, dominating
training over enough steps unless that's actually intended). Pass
`scale_parameter=True, weight_decay=1.0` explicitly to match the legacy
wrapper's defaults.

See each strategy's own module docstring and nodes/smoke_tests/ for what
each one optimizes and its current equivalence/hardware-validation
status. The set of valid `strategy` names lives in one place now,
strategy_registry.py -- see that module's docstring for why.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.adafactor import AdafactorAlgorithm
from .composed import ComposedOptimizerHandle, ParameterGroupPolicy
from .handle import OptimizerHandle
from .node import OptimizerNode
from .strategy_registry import STRATEGIES, STRATEGY_DOC, resolve_strategy


class ComposedAdafactorOptimizerNode(OptimizerNode):
    """Adafactor, composed from a pure-math Algorithm + a selectable
    ExecutionStrategy."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-8, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None,
                     doc="None = Adafactor's own time-varying rho_t schedule for the "
                         "second moment; set for additional first-moment momentum."),
        "scale_parameter": Port(name="scale_parameter", type=bool, required=False, default=False,
                                 doc="True ties the effective step size to the parameter's "
                                     "own current RMS (the legacy default). Has a real "
                                     "failure mode for parameters initialized at/near zero "
                                     "(e.g. LoRA's B matrix): effective step size collapses "
                                     "to roughly 1e-6 * lr and stays there, a self-reinforcing "
                                     "near-standstill. False (default) has no such dependency "
                                     "-- effective step size is just lr."),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0,
                              doc="Decoupled weight decay -- p *= 1 - wd*alpha_t, matching "
                                  "the legacy reference exactly. Legacy default is 1.0, not "
                                  "0.0 -- see module docstring for why this Node defaults "
                                  "conservatively instead."),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "strategy": Port(name="strategy", type=str, required=False, default="simple",
                          choices=tuple(STRATEGIES), doc=STRATEGY_DOC),
        "group_policy": Port(
            name="group_policy", type=ParameterGroupPolicy, required=False, default=None,
            doc="None = UniformGroups (every parameter at the base lr). "
                "LoRAPlusGroups(...) trains LoRA's B matrices at a higher rate than A -- "
                "see nodes/optimizer/composed.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        algorithm = AdafactorAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            beta1=inputs.get("beta1", self.INPUTS["beta1"].default),
            scale_parameter=inputs.get("scale_parameter", self.INPUTS["scale_parameter"].default),
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
