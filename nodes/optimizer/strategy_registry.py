"""Single source of truth for which ExecutionStrategy names are valid,
shared by every Composed*OptimizerNode (composed_adamw.py,
composed_adafactor.py, composed_came.py).

Previously each of those three files declared its own byte-identical
_STRATEGIES dict, its own copy of the same dispatch/validation logic,
and its own copy of the strategy Port's doc string listing the
available names as plain text. Real, working duplication, not a style
nitpick -- it already caused one real bug (shape_grouped registered on
composed_came.py's own copy but not the other two, confirmed and fixed
directly after a real user hit the resulting ValueError on real
hardware) and left a second one live even after that fix: the doc
strings on composed_adafactor.py/composed_adamw.py's own `strategy`
Ports still said "One of 'simple', 'chunked', 'foreach'" after
shape_grouped was added to their dicts, because updating a dict and
updating a hand-written string describing that dict are two separate,
easy-to-forget edits with no single place enforcing they match.
composed_came.py's own doc string happened to stay correct only because
nobody had touched it since shape_grouped was first added there --
not because the duplication was safe.

Centralizing here makes both bug classes structurally impossible going
forward, not just fixed once: one dict, one generated doc string, one
dispatch function, three call sites that can't drift from each other or
from what's actually registered because there's nothing left to copy.

Every ExecutionStrategy here is algorithm-agnostic by construction (see
strategies/base.py) -- there is no case today where one Composed*
node's algorithm needs a *different* set of valid strategy names than
another's, which is what makes one shared registry correct, not just
convenient. A future Algorithm that genuinely couldn't support one of
these would be a first, and should prompt reconsidering this file, not
silently working around it by going back to a per-file copy.
"""

from __future__ import annotations

from .strategies.chunked import ChunkedScratchBufferStrategy
from .strategies.foreach import ForeachApplyStrategy
from .strategies.shape_grouped import ShapeGroupedBatchStrategy
from .strategies.shape_grouped_foreach import ShapeGroupedForeachStrategy
from .strategies.simple import SimpleLoopStrategy

STRATEGIES = {
    "simple": SimpleLoopStrategy,
    "chunked": ChunkedScratchBufferStrategy,
    "foreach": ForeachApplyStrategy,
    "shape_grouped": ShapeGroupedBatchStrategy,
    "shape_grouped_foreach": ShapeGroupedForeachStrategy,
}

DEFAULT_STRATEGY = "shape_grouped_foreach"
"""What every Composed*OptimizerNode's ``strategy`` Port defaults to.

Here rather than written into each of the three nodes for the same reason
``STRATEGY_DOC`` is here: a value three files repeat is three things that
can disagree, and a disagreement about a *default* is invisible -- nothing
fails, two nodes quietly stop behaving like their documented twin.

Measured on the B580, 2026-10-02, ``hw_validate.py main`` with "1024
aes", 100 steps, batch 1, rank 64, seed 1234, one process per run
(steady steps/sec; the losses are the same values across every arm, to
~1.9e-5, as the equivalence smoke tests require):

    adafactor   simple 0.590 -> shape_grouped_foreach 1.007   1.71x
    adamw       simple 0.919 -> shape_grouped_foreach 1.027   1.12x
    came        simple 0.471 -> shape_grouped_foreach 0.972   2.06x

Not the whole matrix, and deliberately so: the three defaults had to be
decided, each was measured against the one thing it was being replaced
by, and a full cross-product would not have changed a decision. What was
worth measuring *was* whether the win was the "foreach" or the
"shape_grouped", since those are separable: shape_grouped alone is
0.991 against foreach alone at 0.574, so the grouping does essentially
all of the work and ``torch._foreach_*`` on its own is slightly slower
than plain ``simple``.

Two caveats that bound this rather than decorate it. It is one parameter
population -- LoRA rank 64 on SDXL, which is the case this optimizer work
targets -- and a different rank or target-module set moves the shape
distribution. And peak reserved memory goes *up* by ~100-200 MB per run,
which matters only if a caller is already at the card's ceiling.

Override per call when a caller needs something else; the port accepts
any name in STRATEGIES.
"""

# Generated from STRATEGIES itself, not hand-written -- cannot list a name
# that isn't (or fail to list one that is) actually registered.
STRATEGY_DOC = (
    f"One of {list(STRATEGIES)} -- see each strategy's own docstring for "
    f"what it optimizes and its current equivalence/hardware-validation status."
)


def resolve_strategy(strategy_name: str):
    """A freshly-constructed ExecutionStrategy for strategy_name, or a
    ValueError listing the real, current set of valid names -- also
    generated from STRATEGIES, so the error message itself can't go
    stale either."""
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy_name!r} -- choose one of {list(STRATEGIES)}"
        )
    return STRATEGIES[strategy_name]()
