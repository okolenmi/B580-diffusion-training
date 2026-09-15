"""ComposedAdamWOptimizerNode: AdamWAlgorithm + a selectable ExecutionStrategy.

The only AdamW node in nodes/optimizer/ -- device-resident, built
entirely from this package's own Algorithm/ExecutionStrategy pieces, no
core.optimizers import. adamw.py's AdamWOptimizerNode (wrapped
core.optimizers.CPUAdamW) and SimpleAdamWOptimizerNode (wrapped
torch.optim.AdamW directly) were removed: CPUAdamW's CPU-resident
optimizer state exists for a full-parameter fine-tune's Adam state,
which genuinely can't fit on the device -- a scenario this project has
no way to produce (every TrainableModel here is LoRA-injected; see
nodes/model/handle.py), so the tradeoff it was built for never
actually applies. SimpleAdamWOptimizerNode's plain device-resident
torch.optim.AdamW is exactly what this Node already does, with more
options (strategy, state_precision, group_policy) on top.

Verified against CPUAdamW's own formula directly (same bias-corrected-lr
AdamW variant, same decoupled-decay-at-base-lr convention) --
see nodes/smoke_tests/smoke_test_adamw_equivalence.py, which still
constructs core.optimizers.CPUAdamW itself as the correctness reference
(core.optimizers is untouched legacy math, not a Node this project
exposes).

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
from .state_store import STATE_PRECISIONS, STATE_PRECISION_DOC, resolve_state_store
from .strategy_registry import STRATEGIES, STRATEGY_DOC, resolve_strategy


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
                          choices=tuple(STRATEGIES), doc=STRATEGY_DOC),
        "group_policy": Port(
            name="group_policy", type=ParameterGroupPolicy, required=False, default=None,
            doc="None = UniformGroups (every parameter at the base lr). "
                "LoRAPlusGroups(...) trains LoRA's B matrices at a higher rate than A -- "
                "see nodes/optimizer/composed.py.",
        ),
        "state_precision": Port(name="state_precision", type=str, required=False,
                                 default="float32", choices=tuple(STATE_PRECISIONS),
                                 doc=STATE_PRECISION_DOC),
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
        state_store = resolve_state_store(
            inputs.get("state_precision", self.INPUTS["state_precision"].default))
        handle = ComposedOptimizerHandle(
            algorithm=algorithm,
            strategy=strategy,
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
            group_policy=inputs.get("group_policy"),
            state_store=state_store,
        )
        result = {"optimizer": handle}
        self.validate_outputs(result)
        return result
