"""ComposedCAMEOptimizerNode: CAMEAlgorithm + SimpleLoopStrategy, composed.

Deliberately a separate class from CAMEOptimizerNode (came.py, which wraps
the old, already-verified-on-real-hardware core.optimizers.ChunkedXPUCAME)
rather than a replacement for it. This composed version's math has been
verified numerically against the same reference (see algorithms/came.py),
but has not been run on real hardware yet, and SimpleLoopStrategy has none
of the memory optimizations (scratch-buffer reuse, MemPool) the legacy
class needs to actually fit in this project's VRAM budget for real training
-- see strategies/simple.py's module docstring. Once a memory-optimized
strategy exists and this composed path has been validated on real hardware,
CAMEOptimizerNode.build() can be pointed at this composition instead of the
legacy wrapper, and the legacy path can eventually be retired -- but that's
a deliberate, separate later step, not this one.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.came import CAMEAlgorithm
from .composed import ComposedOptimizerHandle
from .handle import OptimizerHandle
from .node import OptimizerNode
from .strategies.simple import SimpleLoopStrategy


class ComposedCAMEOptimizerNode(OptimizerNode):
    """CAME, composed from a pure-math Algorithm + the simplest possible
    ExecutionStrategy -- see this module's docstring for status/next steps."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "lr": Port(name="lr", type=float, required=False, default=1e-4),
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-30, 1e-16)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999, 0.9999)),
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        algorithm = CAMEAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
        )
        strategy = SimpleLoopStrategy()
        handle = ComposedOptimizerHandle(
            algorithm=algorithm,
            strategy=strategy,
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
        )
        result = {"optimizer": handle}
        self.validate_outputs(result)
        return result
