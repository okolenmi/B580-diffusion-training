"""ComposedFusedCAMEOptimizerNode: CAMEAlgorithm, executed via backward
hooks through ComposedFusedOptimizerHandle.

No legacy fused CAME class exists to wrap or compare against -- a real
algorithm-engineering gap in core/optimizers.py, not something this
adapter layer was ever going to unlock by itself. What this Node
demonstrates instead: ComposedFusedOptimizerHandle is genuinely
algorithm-agnostic, not secretly Adafactor-shaped -- CAMEAlgorithm plugs
into it with zero changes to composed_fused.py, exactly as
composed_fused_adafactor.py does. No independent formula verification
needed here beyond that: CAMEAlgorithm's own math is already verified
elsewhere (algorithms/came.py); what's specific to *this* Node is only
that the hook-driven execution model applies it correctly, which
smoke_test_composed_fused_came.py checks the same way
smoke_test_composed_fused_adafactor.py does for Adafactor.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.came import CAMEAlgorithm
from .composed_fused import ComposedFusedOptimizerHandle
from .handle import FusedOptimizerHandle
from .node import OptimizerNode


class ComposedFusedCAMEOptimizerNode(OptimizerNode):
    """CAME, fused into backward-pass hooks via ComposedFusedOptimizerHandle."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "lr": Port(name="lr", type=float, required=False, default=1e-4),
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-30, 1e-16)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999, 0.9999)),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0),
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
        algorithm = CAMEAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
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
