"""ComposedAdafactorOptimizerNode: AdafactorAlgorithm + a selectable
ExecutionStrategy.

Device-resident, built entirely from this package's own Algorithm/
ExecutionStrategy pieces, no core.optimizers import. As of 2026-10-02
this is the only Adafactor node in the graph: adafactor.py's
AdafactorOptimizerNode (a pass-through to
core.optimizers.ChunkedXPUAdafactor) is deleted, which makes this the
same kind of single-replacement retirement CAMEOptimizerNode already
had -- with one difference worth stating plainly rather than glossing.

**What retiring it gave up, and why that was the right trade.**
ChunkedXPUAdafactor routes every parameter under 10,000 elements through
a cross-parameter fast path: it concatenates *all* of them into one flat
tensor sharing a single clip threshold and a single second-moment state.
Two things follow. First, it is a batching-strategy concern rather than a
per-parameter algorithm one, so no Algorithm could implement it --
AdafactorAlgorithm structurally cannot see other parameters in the same
optimizer (see algorithms/base.py). Second, and the reason this was
retired rather than reimplemented, the shared state **contaminates**:
nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py Part C
measured that a parameter's update depends on unrelated parameters'
gradients sharing its batch, which is why that script tests only
parameters >= 10,000 elements for equivalence with the per-parameter
Algorithm. Reproducing that would mean deliberately reintroducing
cross-parameter coupling into the canonical math.

The formula gap that *was* real is closed: AdafactorAlgorithm implements
FusedXPUAdafactor's tiny-parameter branch (plain elementwise second
moment instead of row/col factoring) via the opt-in
tiny_parameter_threshold -- see _is_factored()'s own docstring. This Node
still doesn't pass that threshold, because doing so would only be correct
against FusedXPUAdafactor, and this Node's reference was Chunked, whose
tiny-parameter mechanism is the unrelated cross-parameter one above.

**The honest residue.** That leaves one unmeasured trade: Chunked's
concatenation collapsed hundreds of small parameters' clip/EMA/normalize
into one kernel launch each, and no strategy here reproduces exactly
that. strategy="foreach" recovers much of the launch-overhead win via
torch._foreach_* without the contamination, and is the setting to reach
for on a graph with many small LoRA matrices. The remaining delta has not
been quantified on real hardware -- see docs/known-issues/open.md for
the measurement that would settle it. Writing an unbenchmarked batching
strategy to close it was the alternative, and would have violated this
project's own rule that a technique earns a place only with real
evidence.

FusedXPUAdafactor (composed_fused_adafactor.py's equivalent concern) has
its own, genuinely per-parameter tiny-parameter mechanism.
ComposedFusedAdafactorOptimizerNode is the only Node that sets
tiny_parameter_threshold. ForeachXPUAdafactor (formerly
foreach_adafactor.py) had no tiny-parameter special case at all,
confirmed the same way -- deleted 2026-09-16, the same way
CAMEOptimizerNode was once its own equivalence was established;
ComposedAdafactorOptimizerNode(strategy="foreach") is the only way to get
foreach-strategy Adafactor.

`INPUTS` below default to the conservative, predictable values
(`scale_parameter=False, weight_decay=0.0`) rather than the deleted
AdafactorOptimizerNode's own defaults (`scale_parameter=True,
weight_decay=1.0`): those were unusual (full weight decay of 1.0 shrinks
any parameter by ~5% per step at a typical lr, dominating training over
enough steps unless that's actually intended). Pass
`scale_parameter=True, weight_decay=1.0` explicitly to reproduce them.

scale_parameter=True also has a documented pathology independent of the
weight decay, spelled out in AdafactorAlgorithm's own module docstring:
for a parameter initialized at or near zero -- LoRA's B matrix is zero by
convention -- the effective step size collapses to roughly 1e-6 * lr and
stays there. Both this implementation and the reference it replaced
reproduce that identically; it is a property of the algorithm, not of
this Node.

See each strategy's own module docstring and nodes/smoke_tests/ for what
each one optimizes and its current equivalence/hardware-validation
status. The set of valid `strategy` names lives in one place now,
strategy_registry.py -- see that module's docstring for why.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .algorithms.adafactor import AdafactorAlgorithm
from .composed import ComposedOptimizerHandle, ParameterGroupPolicy
from .handle import OptimizerHandle
from .node import OptimizerNode
from .state_store import STATE_PRECISIONS, STATE_PRECISION_DOC, resolve_state_store
from .strategy_registry import STRATEGIES, STRATEGY_DOC, resolve_strategy


class ComposedAdafactorOptimizerNode(OptimizerNode):
    """Adafactor, composed from a pure-math Algorithm + a selectable
    ExecutionStrategy."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-8, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None,
                     doc="None = Adafactor's own time-varying rho_t schedule for the "
                         "second moment; set for additional first-moment momentum."),
        "scale_parameter": Port(name="scale_parameter", type=bool, required=False, default=False,
                                 doc="True ties the effective step size to the parameter's "
                                     "own current RMS (the legacy default). Has a real "
                                     "failure mode for parameters initialized at/near zero "
                                     "(e.g. LoRA's B matrix): effective step size collapses "
                                     "to roughly 1e-6 * lr and stays there, a self-reinforcing "
                                     "near-standstill. False (default) has no such dependency "
                                     "-- effective step size is just lr."),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0,
                              doc="Decoupled weight decay -- p *= 1 - wd*alpha_t, matching "
                                  "the legacy reference exactly. Legacy default is 1.0, not "
                                  "0.0 -- see module docstring for why this Node defaults "
                                  "conservatively instead."),
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
        algorithm = AdafactorAlgorithm(
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            beta1=inputs.get("beta1", self.INPUTS["beta1"].default),
            scale_parameter=inputs.get("scale_parameter", self.INPUTS["scale_parameter"].default),
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
