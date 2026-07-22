"""ComposedCAMEOptimizerNode: CAMEAlgorithm + a selectable ExecutionStrategy.

Deliberately a separate class from CAMEOptimizerNode (came.py, which wraps
the old, already-verified-on-real-hardware core.optimizers.ChunkedXPUCAME)
rather than a replacement for it.

The `strategy` input is where the Algorithm/ExecutionStrategy split
becomes real, usable value rather than just a design claim: switching
"simple" <-> "chunked" changes nothing about CAMEAlgorithm at all, only
how ExecutionStrategy iterates parameters and manages temporary memory --
confirmed identical training behavior between the two (bit-exact match
across 60 steps on two differently-shaped parameters, see this session's
verification history), so choosing between them is purely a memory/
performance decision, never a correctness one.

Status of each combination, stated precisely:
  - "simple" (default): validated end-to-end on real XPU hardware
    (a genuine toy-regression training run, plus every lifecycle method
    including a real offload/reload device round trip -- all passed).
  - "chunked": verified to produce bit-identical training results to
    "simple" via a numpy-backed equivalence test, and its MemoryManager-
    backed scratch buffer's cross-step caching + offload cleanup are
    verified end-to-end -- both via real torch on CPU and, since,
    confirmed passing on real XPU hardware by the user (97.7% loss
    reduction, all lifecycle methods, offload/reload round trip, and the
    caching/cleanup check all passed -- see
    smoke_test_composed_came.py's check [4]). Real memory savings are
    partial and precisely scoped -- see strategies/chunked.py's module
    docstring for exactly what it does and doesn't optimize yet (no
    MemPool integration, and CAMEAlgorithm doesn't yet use the scratch
    hint for its own internal intermediates).

Once "chunked" (or a further-optimized successor) is real-hardware
validated and matches core/optimizers.py's ChunkedXPUCAME on actual VRAM
usage, CAMEOptimizerNode.build() can be pointed at this composition
instead of the legacy wrapper, and the legacy path can eventually be
retired -- a deliberate, separate later step, not this one.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.came import CAMEAlgorithm
from .composed import ComposedOptimizerHandle
from .handle import OptimizerHandle
from .node import OptimizerNode
from .strategies.simple import SimpleLoopStrategy
from .strategies.chunked import ChunkedScratchBufferStrategy

_STRATEGIES = {
    "simple": SimpleLoopStrategy,
    "chunked": ChunkedScratchBufferStrategy,
}


class ComposedCAMEOptimizerNode(OptimizerNode):
    """CAME, composed from a pure-math Algorithm + a selectable
    ExecutionStrategy -- see this module's docstring for per-strategy
    validation status."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "lr": Port(name="lr", type=float, required=False, default=1e-4),
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-30, 1e-16)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999, 0.9999)),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "strategy": Port(name="strategy", type=str, required=False, default="simple",
                          doc="'simple' (real-hardware validated) or 'chunked' "
                              "(equivalence-verified, not yet real-hardware validated)."),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        algorithm = CAMEAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
        )
        strategy_name = inputs.get("strategy", self.INPUTS["strategy"].default)
        if strategy_name not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy_name!r} -- choose one of {list(_STRATEGIES)}"
            )
        strategy = _STRATEGIES[strategy_name]()
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
