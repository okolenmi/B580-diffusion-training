"""ComposedFusedAdafactorOptimizerNode: AdafactorAlgorithm, executed via
backward hooks through ComposedFusedOptimizerHandle.

Same relationship to FusedAdafactorOptimizerNode (fused_adafactor.py,
which wraps the legacy core.optimizers.FusedXPUAdafactor) as
ComposedAdafactorOptimizerNode has to AdafactorOptimizerNode -- adds a
non-legacy alternative, doesn't touch or retire the legacy wrapper.

**Now matches FusedXPUAdafactor's small-parameter (< 10,000 element)
formula too, not just its large-parameter one.** Was a real, documented
gap (see docs/CLEANUP_TODO.md for the full history): FusedXPUAdafactor
has a TINY_NUMEL special case that swaps in a full elementwise
second-moment buffer instead of the row/col factored approximation for
small parameters -- a real formula change, not just a storage-layout
optimization (see composed_fused.py's module docstring). Closed by
passing tiny_parameter_threshold=10_000 to AdafactorAlgorithm below,
which already had everywhere it needed for this (the existing
elementwise "vs" state/update path already handles any shape, not just
1D -- see AdafactorAlgorithm._is_factored()'s own docstring for exactly
what changed and the one deliberate, small, documented simplification
this doesn't chase (a first-update-only ~1e-4-relative difference from
not replicating FusedXPUAdafactor's lazy initialization quirk exactly).
Verification: nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py
was extended with a Part D specifically for this -- run it and confirm
before trusting this over the legacy node for real training.
ChunkedXPUAdafactor's own, different tiny-parameter mechanism (cross-
parameter batching, not per-parameter) is unrelated to this fix and
still open -- see ComposedAdafactorOptimizerNode's own docstring.
Separately, FusedXPUAdafactor has a real, confirmed momentum-corruption
bug for float32 parameters (docs/known-issues/open.md) that this Node
does not reproduce.
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
    module's docstring for how small-parameter handling was brought in
    line with the legacy reference."""

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
            tiny_parameter_threshold=10_000,  # matches FusedXPUAdafactor's TINY_NUMEL
            # exactly -- this Node specifically, not ComposedAdafactorOptimizerNode,
            # since Foreach/Chunked's own tiny-parameter behavior differs (see
            # AdafactorAlgorithm._is_factored()'s docstring).
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
