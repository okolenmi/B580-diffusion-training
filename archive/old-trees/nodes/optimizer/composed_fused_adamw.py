"""ComposedFusedAdamWOptimizerNode: AdamWAlgorithm, executed via backward
hooks through ComposedFusedOptimizerHandle.

Same "no legacy fused reference exists, this demonstrates genuine
algorithm-agnosticism rather than replacing anything" position as
composed_fused_came.py -- see that module's docstring.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.adamw import AdamWAlgorithm
from .composed_fused import ComposedFusedOptimizerHandle
from .handle import FusedOptimizerHandle
from .node import OptimizerNode


class ComposedFusedAdamWOptimizerNode(OptimizerNode):
    """AdamW, fused into backward-pass hooks via ComposedFusedOptimizerHandle."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999)),
        "eps": Port(name="eps", type=float, required=False, default=1e-8),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=1e-2),
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "optimizer": Port(
            name="optimizer", type=FusedOptimizerHandle, required=True,
            doc="A constructed, ready-to-use fused (backward-hook-based) optimizer. "
                "Hooks are already registered -- call begin_step() before backward(), "
                "not step() (which is a no-op, see ComposedFusedOptimizerHandle).",
        ),
    }

    def build(self, **inputs) -> dict[str, FusedOptimizerHandle]:
        self.validate_inputs(inputs)
        algorithm = AdamWAlgorithm(
            betas=inputs.get("betas", self.INPUTS["betas"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
        )
        handle = ComposedFusedOptimizerHandle(
            algorithm=algorithm,
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
        )
        result = {"optimizer": handle}
        self.validate_outputs(result)
        return result
