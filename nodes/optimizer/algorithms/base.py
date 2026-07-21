"""Algorithm: pure per-parameter update math.

Knows the shape of persistent state a single parameter needs, and how to
turn a gradient + that state into an update -- nothing else. In particular:
no knowledge of GPU memory management, scratch buffers, torch._foreach_*
vectorization, or backward hooks. That separation is the entire point: it's
what turns "run algorithm X under execution strategy Y" into a composition
of two independently-written, independently-testable pieces instead of an
M-algorithms x N-strategies grid of hand-written classes (which is what
core/optimizers.py's ChunkedXPUAdafactor/ChunkedXPUCAME/ForeachXPUAdafactor/
FusedXPUAdafactor actually are, on inspection -- 2 algorithms x up to 3
memory strategies, hand-crossed, with CAME only getting 1 of the 3 possible
strategies because writing each combination by hand is expensive). See
docs/nodes_package_design.md's "Algorithm/ExecutionStrategy separation"
section for the full reasoning, including why the tiny-parameter batching
trick some of the old classes use is a strategy concern, not an algorithm
one -- it changes how state is *allocated/batched* for many small
parameters, never what update *formula* gets computed.

lr is deliberately NOT known to Algorithm at all -- an ExecutionStrategy
applies `param -= lr * update` (or with per-parameter-group lr, or however
else it wants), so an Algorithm never needs to know or care about learning
rate, only about turning (grad, state) into an update.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Algorithm(ABC):

    @abstractmethod
    def init_state(self, param_shape, dtype, device) -> dict[str, Any]:
        """Zero-initialized per-parameter state for a parameter of the
        given shape. Returns a plain dict of named tensors -- kept as a
        dict (not a per-algorithm dataclass) specifically so an
        ExecutionStrategy can manage state generically (iterate values for
        offload/reload/decay/reset) without needing to know anything
        algorithm-specific about what's inside."""

    @abstractmethod
    def compute_update(self, grad, state: dict[str, Any], scratch=None):
        """Given the current gradient and this parameter's state (mutated
        in place as needed), return the update to subtract from the
        parameter. Not lr-scaled -- see module docstring.

        scratch: optional tensor, same shape as grad, that an
        ExecutionStrategy may provide as reusable workspace (e.g. a single
        buffer shared across all parameters in a step, avoiding N separate
        temporary allocations). Purely an optional hint -- an Algorithm is
        free to ignore it and allocate normally (correct, just not
        maximally memory-efficient), or to use it for its own internal
        intermediates via in-place ops. This is deliberately NOT the same
        thing as "avoid allocating N per-parameter scratch buffers instead
        of one shared one" (an ExecutionStrategy concern, doesn't need
        Algorithm cooperation at all) -- it's specifically for an Algorithm
        that wants to restructure its *own* internal computation (e.g.
        writing successive intermediates into the same buffer rather than
        allocating a fresh tensor per intermediate step) to reduce peak
        memory further. See docs/nodes_package_design.md's follow-up notes
        on this distinction, and this session's earlier, carefully-verified
        buffer-reuse fix in core/optimizers.py's ChunkedXPUCAME for what
        real in-place restructuring looks like when done correctly.
        """

    @abstractmethod
    def decay_state(self, state: dict[str, Any], factor: float) -> None:
        """Scale state in place by factor. factor<=0 should behave like a
        full reset_state()."""

    @abstractmethod
    def reset_state(self, state: dict[str, Any]) -> None:
        """Reset state in place to its zero-initialized values."""
