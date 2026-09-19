"""ComposedFusedAdafactorOptimizerNode: AdafactorAlgorithm, executed via
backward hooks through ComposedFusedOptimizerHandle.

The only fused Adafactor node in nodes/optimizer/ now -- the legacy
fused_adafactor.py (FusedAdafactorOptimizerNode, wrapped
core.optimizers.FusedXPUAdafactor) was deleted once this Node was
confirmed equivalent, including for small (< 10,000 element)
parameters -- see below (docs/CLEANUP_TODO.md, where this was tracked
during development, has since been deleted; the full history is in git
log for this file and nodes/optimizer/algorithms/adafactor.py).

**Matches FusedXPUAdafactor's small-parameter formula, not just its
large-parameter one -- confirmed on real torch, not just reasoned
about.** FusedXPUAdafactor has a TINY_NUMEL special case that swaps in
a full elementwise second-moment buffer instead of the row/col factored
approximation for small parameters -- a real formula change, not just a
storage-layout optimization (see composed_fused.py's module docstring).
Closed by passing tiny_parameter_threshold=10_000 to AdafactorAlgorithm
below, which needed less new code than expected: the existing
elementwise "vs" state/update path already handles any shape, not just
1D -- see AdafactorAlgorithm._is_factored()'s own docstring for exactly
what changed and the one deliberate, small, documented simplification
it doesn't chase (a first-update-only ~1e-4-relative difference from
not replicating FusedXPUAdafactor's lazy initialization quirk exactly,
confirmed negligible by the same real-torch run below). Permanent
regression coverage lives in
nodes/smoke_tests/smoke_test_fused_adafactor_equivalence.py, alongside
the rest of this pair's equivalence checks -- the investigation itself,
including the two numbers that initially looked like problems but
turned out to already be inside this project's own established
tolerances (see that file's _TOLERANCES/_TINY_TOLERANCES), is preserved
in nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py's Part D
writeup.
ChunkedXPUAdafactor's own, different tiny-parameter mechanism (cross-
parameter batching, not per-parameter) is unrelated to this fix and
still open -- see docs/design/09-prioritized-backlog.md.
Separately, FusedXPUAdafactor had a real momentum-corruption
bug for float32 parameters, fixed at the source
(core/optimizers.py) and confirmed by user -- see
docs/known-issues/resolved.md -- that this Node never reproduced even
before that fix (its own momentum blend was always correct; only the
legacy reference was wrong).
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
