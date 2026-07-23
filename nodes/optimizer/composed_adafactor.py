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
shaped algorithm. See `algorithms/adafactor.py`'s module docstring for
the full verification writeup.

**`scale_parameter` and `weight_decay` are now both implemented and
verified** (an earlier version of this Node fixed both off, pending a
real interface extension -- see `algorithms/base.py`'s module docstring
for that extension). Both `INPUTS` below default to the conservative,
predictable values (`scale_parameter=False, weight_decay=0.0`) rather
than `AdafactorOptimizerNode`'s own legacy defaults
(`scale_parameter=True, weight_decay=1.0`) -- a deliberate choice, not an
oversight: those legacy defaults are unusual (full weight decay of 1.0
shrinks any parameter by ~5% per step at a typical lr, dominating
training over enough steps unless that's actually intended), and
defaulting this Node to match them would have silently changed already-
tested toy-regression behavior for anyone not paying close attention.
Pass `scale_parameter=True, weight_decay=1.0` explicitly to match the
legacy wrapper's defaults exactly -- verified to do so, see
`algorithms/adafactor.py`.

**A real, precisely-characterized pathology worth knowing before turning
`scale_parameter=True` on**: its effective step size is
`clamp(param_rms**2, min=~1e-6) * lr` -- for a parameter initialized at
or near zero (LoRA's B matrix, by convention, is initialized to exactly
zero), this collapses to roughly `1e-6 * lr`, about a millionth of the
plain step size, and stays there indefinitely: near-zero updates keep
the parameter near zero, which keeps `alpha_t` near the floor -- a
self-reinforcing near-standstill, not a bug in this implementation or
the legacy reference (both reproduce it identically, verified directly
against `core.optimizers.ChunkedXPUAdafactor`). This is very likely the
explanation for training appearing to make near-zero progress under
`scale_parameter=True` in practice. `scale_parameter=False` (this Node's
default) doesn't have this failure mode at all -- effective step size is
just `lr`, independent of the parameter's own magnitude.

Status of each strategy, stated precisely:
  - "simple": equivalence-verified against core.optimizers.ChunkedXPUAdafactor
    directly (real torch, CPU), across scale_parameter on/off, weight
    decay on/off, momentum on/off, factored and non-factored parameters,
    float32 and bf16 -- see algorithms/adafactor.py's module docstring
    for the full breakdown and numbers. Not yet run on real XPU hardware
    (unlike ComposedCAMEOptimizerNode's "simple", which the user has
    validated there).
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
    """Adafactor, composed from a pure-math Algorithm + a selectable
    ExecutionStrategy. scale_parameter and weight_decay both fully
    supported and verified -- see module docstring for their (safe,
    conservative) defaults and why they differ from the legacy wrapper's."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-8, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None,
                     doc="None = Adafactor's own time-varying rho_t schedule for the "
                         "second moment; set for additional first-moment momentum."),
        "scale_parameter": Port(name="scale_parameter", type=bool, required=False, default=False,
                                 doc="True ties the effective step size to the parameter's "
                                     "own current RMS (the legacy default) -- has a real "
                                     "failure mode for parameters initialized at/near zero "
                                     "(e.g. LoRA's B matrix). See module docstring before "
                                     "enabling."),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0,
                              doc="Decoupled weight decay -- p *= 1 - wd*alpha_t, matching "
                                  "the legacy reference exactly. Legacy default is 1.0, not "
                                  "0.0 -- see module docstring for why this Node defaults "
                                  "conservatively instead."),
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
            scale_parameter=inputs.get("scale_parameter", self.INPUTS["scale_parameter"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
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
