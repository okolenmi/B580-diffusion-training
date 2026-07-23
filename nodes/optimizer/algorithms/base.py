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

**Revised design decision (this contract was extended once already --
here's why, stated precisely):** an earlier version of this docstring
said lr was deliberately unknown to Algorithm, with an ExecutionStrategy
applying `param -= lr * update` uniformly. That held for CAME, but broke
down for real when `AdafactorAlgorithm` was built: Adafactor's
`scale_parameter` mode computes its effective step size from
`clamp(param_rms**2, min) * lr` -- genuinely dependent on both `lr` and
the *live parameter's own current magnitude* -- and its weight decay
(`p *= 1 - wd*alpha_t`) is a *multiplicative* rescale of the live
parameter, coupled to that same `alpha_t`, which no additive delta could
express regardless of what `compute_update()` receives as input.

So `compute_update()` now receives `param` (read-only access to the
parameter's current value) and `lr`, and returns `(delta, decay)`:
`delta` is the final, already-lr-scaled amount to subtract, and `decay`
is either `None` or a multiplicative factor an ExecutionStrategy applies
to `param.data` *before* subtracting `delta` -- matching the order every
legacy optimizer in `core/optimizers.py` actually uses (decay first,
then the additive step). `Algorithm` itself never mutates `param` --
`decay` is a description of what to do, not an action taken directly --
keeping the "pure math, `state` is the only thing mutated in place"
property intact even though `param` is now visible to it.

This is a genuine, load-bearing generalization, not a speculative one:
`CAMEAlgorithm` didn't strictly need `param`/`lr`/`decay` for its own
update math, but folding `lr` into its own return value and adding
`weight_decay` support through the exact same `decay` mechanism was
close to free once the contract existed -- real, working feature parity
CAME's port didn't have before (see `algorithms/came.py`). One universal
contract, not a special case bolted on for Adafactor alone.
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
    def compute_update(self, grad, param, state: dict[str, Any], lr: float, scratch=None):
        """Given the current gradient, this parameter's live value, this
        parameter's state (mutated in place as needed), and the current
        learning rate, return `(delta, decay)`:

        - `delta`: the final, already-lr-scaled amount to subtract from
          `param.data`. Not a "unit" update needing external scaling --
          see module docstring for why lr moved into this contract.
        - `decay`: `None`, or a multiplicative factor to apply to
          `param.data` *before* `delta` is subtracted (decoupled weight
          decay, or anything else that's a rescale of the parameter's
          current value rather than an additive step). An Algorithm
          computes this value but never applies it -- an ExecutionStrategy
          does `if decay is not None: param.data.mul_(decay)` before
          `param.data.sub_(delta)`, matching every legacy optimizer's own
          order of operations in core/optimizers.py.

        `param`: read-only. An Algorithm may read `param.data` (e.g. for
        Adafactor's `scale_parameter`, which needs the parameter's own
        current RMS) but must never mutate it directly -- `decay` is how
        an Algorithm expresses "rescale the parameter," not direct
        mutation, so this class stays pure math with `state` (and,
        optionally, `scratch`'s contents) as the only things it mutates.

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
