"""ComposedFusedAdafactorOptimizerNode: AdafactorAlgorithm, executed via
backward hooks through ComposedFusedOptimizerHandle.

Same relationship to FusedAdafactorOptimizerNode (fused_adafactor.py,
which wraps the legacy core.optimizers.FusedXPUAdafactor) as
ComposedAdafactorOptimizerNode has to AdafactorOptimizerNode -- adds a
non-legacy alternative, doesn't touch or retire the legacy wrapper.

**Not bit-exact with FusedXPUAdafactor for every parameter, and that's a
deliberate, documented scope decision, not an oversight:** FusedXPUAdafactor
has a TINY_NUMEL (10,000-element) special case that swaps in a full
elementwise second-moment buffer instead of the row/col factored
approximation for small parameters -- confirmed by reading
FusedXPUAdafactor._update_param directly to be a real formula change, not
just a storage-layout optimization (see composed_fused.py's module
docstring, which also corrects an earlier, less precise characterization
of this same trick in algorithms/base.py's docstring). AdafactorAlgorithm
doesn't implement that branch -- extending it to is real, separate
algorithm-engineering work, not a fused-execution concern, so it's left
for later rather than bolted on here to chase bit-exactness. Practical
consequence: for parameters under 10,000 elements (most individual LoRA
matrices), this Node's math differs from FusedXPUAdafactor's; for larger
parameters, both branches already agree (see algorithms/adafactor.py),
and that's exactly the regime smoke_test_fused_adafactor_equivalence.py
checks.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.adafactor import AdafactorAlgorithm
from .composed_fused import ComposedFusedOptimizerHandle
from .handle import FusedOptimizerHandle
from .node import OptimizerNode


class ComposedFusedAdafactorOptimizerNode(OptimizerNode):
    """Adafactor, fused into backward-pass hooks via ComposedFusedOptimizerHandle
    -- see that module's docstring for the execution model and this
    module's docstring for the one documented divergence from the legacy
    reference (small parameters, see above)."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-8, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None,
                     doc="None = Adafactor's own time-varying rho_t schedule for the "
                         "second moment; set for additional first-moment momentum."),
        "scale_parameter": Port(name="scale_parameter", type=bool, required=False, default=False,
                                 doc="See ComposedAdafactorOptimizerNode's docstring for why "
                                     "this defaults to False, not the legacy True."),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0,
                              doc="See ComposedAdafactorOptimizerNode's docstring for why "
                                  "this defaults to 0.0, not the legacy 1.0."),
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
        algorithm = AdafactorAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            beta1=inputs.get("beta1", self.INPUTS["beta1"].default),
            scale_parameter=inputs.get("scale_parameter", self.INPUTS["scale_parameter"].default),
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
