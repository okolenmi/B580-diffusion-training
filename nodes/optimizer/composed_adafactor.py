"""ComposedAdafactorOptimizerNode: AdafactorAlgorithm + a selectable
ExecutionStrategy.

Same relationship to AdafactorOptimizerNode (adafactor.py, which wraps
the legacy core.optimizers.ChunkedXPUAdafactor) as ComposedCAMEOptimizerNode
has to CAMEOptimizerNode -- see that module's docstring for the pattern.

This is the second Algorithm built in this package, and this Node is
where the Algorithm/ExecutionStrategy split's generalization actually
gets exercised end-to-end: the same `_STRATEGIES` dict, same
`SimpleLoopStrategy`/`ChunkedScratchBufferStrategy` classes as
`composed_came.py`, completely unmodified, now driving a differently-
shaped algorithm (time-varying `rho_t` schedule instead of fixed EMA
betas, a `begin_step()` hook CAME never needed, different per-parameter
math entirely). See `algorithms/adafactor.py`'s module docstring for the
full verification writeup, including a real, precisely-scoped limitation
this composition does NOT cover yet.

**Real, precisely-scoped limitation -- this is NOT a drop-in replacement
for AdafactorOptimizerNode's default configuration.** The legacy wrapper
defaults to `scale_parameter=True, weight_decay=1.0` -- this Node only
implements the `scale_parameter=False, weight_decay=0` case (see
`algorithms/adafactor.py`'s module docstring for exactly why: both need
`compute_update()` to know `lr` and/or the live parameter's own current
magnitude, which the existing Algorithm contract doesn't provide -- a
real, separate interface extension, not attempted here). Both INPUTS
below are consequently fixed, not exposed as configurable ports, so a
caller can't accidentally ask this Node for behavior it doesn't actually
implement.

Status of each strategy, stated precisely:
  - "simple": equivalence-verified against core.optimizers.ChunkedXPUAdafactor
    directly (real torch, CPU) -- see algorithms/adafactor.py's module
    docstring. Not yet run on real XPU hardware (unlike
    ComposedCAMEOptimizerNode's "simple", which the user has validated
    there).
  - "chunked": same equivalence verification, same "not yet run on real
    XPU hardware" status. Its MemoryManager-backed scratch-buffer
    behavior is already covered by strategies/chunked.py's own tests
    (algorithm-agnostic), not re-tested here.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.adafactor import AdafactorAlgorithm
from .composed import ComposedOptimizerHandle
from .handle import OptimizerHandle
from .node import OptimizerNode
from .strategies.simple import SimpleLoopStrategy
from .strategies.chunked import ChunkedScratchBufferStrategy

_STRATEGIES = {
    "simple": SimpleLoopStrategy,
    "chunked": ChunkedScratchBufferStrategy,
}


class ComposedAdafactorOptimizerNode(OptimizerNode):
    """Adafactor (scale_parameter=False, weight_decay=0 only -- see
    module docstring), composed from a pure-math Algorithm + a
    selectable ExecutionStrategy."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-8, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None,
                     doc="None = Adafactor's own time-varying rho_t schedule for the "
                         "second moment; set for additional first-moment momentum."),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "strategy": Port(name="strategy", type=str, required=False, default="simple",
                          doc="'simple' or 'chunked' -- both equivalence-verified "
                              "against the legacy reference, neither yet run on real "
                              "XPU hardware. See this module's docstring."),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        algorithm = AdafactorAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            beta1=inputs.get("beta1", self.INPUTS["beta1"].default),
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
