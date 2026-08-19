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
from .strategies.simple import SimpleLoopStrategy

STRATEGIES = {
    "simple": SimpleLoopStrategy,
    "chunked": ChunkedScratchBufferStrategy,
    "foreach": ForeachApplyStrategy,
    "shape_grouped": ShapeGroupedBatchStrategy,
}

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
