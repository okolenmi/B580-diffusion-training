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

**Real limitation found while building AdafactorAlgorithm, stated
precisely rather than papered over:** the claim above holds for CAME, but
not unconditionally for every algorithm. Adafactor's reference
implementation (core/optimizers.py's ChunkedXPUAdafactor) has a
`scale_parameter` mode where the effective step size is
`clamp(param_rms**2, min) * lr` -- genuinely dependent on both `lr` and
the *live parameter's own current magnitude*, neither of which
`compute_update(grad, state, scratch)` has access to. That mode, and
Adafactor's coupled weight-decay (`p *= 1 - wd*alpha_t`, itself
`alpha_t`-dependent), are real, out of scope for AdafactorAlgorithm's
first slice -- see `algorithms/adafactor.py`'s module docstring for
exactly what's deferred and why, and
`docs/nodes_package_design.md`'s "AdafactorAlgorithm" section for the
fuller reasoning. The `scale_parameter=False` case, however, reduces to
`alpha_t = max(eps1, 1.0) * lr`, which for any realistic `eps1 < 1` is
just `lr` -- so it fits this contract exactly as written, no change
needed for that case.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Algorithm(ABC):

    def begin_step(self, n_steps: int = 1) -> None:
        """Called once per real optimizer step, *before* compute_update()
        runs for any parameter -- not once per parameter. Default no-op:
        most algorithms (CAME, with its fixed EMA betas) have no
        once-per-step-not-once-per-parameter bookkeeping to do. Exists
        because Adafactor's `rho_t` schedule is a genuine counterexample:
        it depends on a single, monotonically increasing step counter
        shared across every parameter in a step, which compute_update()
        (called once per parameter) has no way to update exactly once per
        step on its own -- see algorithms/adafactor.py's begin_step() for
        the concrete case this exists for. Every ExecutionStrategy calls
        this exactly once at the top of step(), before its per-parameter
        loop -- see strategies/simple.py or strategies/chunked.py.
        """

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
