"""Algorithm: pure per-parameter update math.

Knows the shape of persistent state a single parameter needs, and how to
turn a gradient + that state into an update -- nothing else. In
particular: no knowledge of GPU memory management, scratch buffers,
torch._foreach_* vectorization, or backward hooks. That separation is
what turns "run algorithm X under execution strategy Y" into a
composition of two independently-written, independently-testable pieces
instead of an M-algorithms x N-strategies grid of hand-written classes.

compute_update() takes `param` (read-only) and `lr`, and returns
`(delta, decay)`: `delta` is the final, already-lr-scaled amount to
subtract, and `decay` is either `None` or a multiplicative factor an
ExecutionStrategy applies to `param.data` *before* subtracting `delta`
(decoupled weight decay, or anything else that rescales the parameter's
current value rather than adding to it). `Algorithm` itself never
mutates `param` directly -- `decay` describes what to do, an
ExecutionStrategy is what does it -- keeping "pure math, `state` is the
only thing mutated in place" intact even though `param` is visible.
`param`/`lr` are both needed by real algorithms: Adafactor's
`scale_parameter` mode computes its effective step size from
`clamp(param_rms**2, min) * lr`, genuinely dependent on both `lr` and the
live parameter's own current magnitude, and its weight decay
(`p *= 1 - wd*alpha_t`) is a multiplicative rescale no additive delta
alone could express.
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
        memory further.
        """

    def compute_update_batched(self, grad_stack, params: list, states: list[dict],
                                lr: float):
        """Same contract as compute_update(), but for a *group* of k
        parameters that share an identical shape (and, critically, an
        identical lr): `grad_stack` is `(k, *shape)`, `params`/`states`
        are length-k lists (states NOT pre-stacked -- exactly how state is
        laid out for batched math is algorithm-specific, so an override is
        responsible for its own stacking; see algorithms/came.py's
        override for the actual pattern: stack fresh from each member's
        existing dict, compute, scatter results back into those same
        dicts, so ComposedOptimizerHandle's state lifecycle -- offload/
        reload/decay/reset, all written once generically over a flat list
        of per-parameter dicts -- needs no changes at all). Returns
        `(delta_stack, decay)`: `delta_stack` is `(k, *shape)`, `decay` is
        shared across the whole group (`None`, or a single float -- true
        by construction whenever wd/lr are the algorithm-level and
        group-level constants they currently are for every Algorithm this
        contract has; an Algorithm whose decay genuinely varied per group
        member would need this contract extended again).

        Default implementation: loops calling compute_update() once per
        group member and stacks the results -- correct for *any*
        Algorithm satisfying the base contract, just without the actual
        batching speedup. Override this (as CAMEAlgorithm does) only once
        real multi-tensor batched math for that algorithm has been
        written and numerically verified -- see algorithms/came.py's own
        override.

        **Real bug this used to have, not just a documented risk: this
        default silently kept only the *last* group member's decay when
        decay actually varied across the group** -- exactly
        AdafactorAlgorithm's scale_parameter=True case (alpha_t, and
        therefore decay, depends on each parameter's own norm). Found by
        a real equivalence-test failure
        (smoke_test_shape_grouped_foreach_equivalence.py), not by
        inspection -- an earlier test of the same fallback path happened
        to use weight_decay=0.0, where decay is always None regardless
        of alpha_t, silently avoiding the bug rather than proving its
        absence. Now detected and raised explicitly instead of silently
        applying the wrong decay to every member but the last."""
        deltas = []
        decays = []
        for i in range(grad_stack.shape[0]):
            d, member_decay = self.compute_update(grad_stack[i], params[i], states[i], lr)
            deltas.append(d)
            decays.append(member_decay)
        import torch

        first_decay = decays[0]
        if any(d != first_decay for d in decays[1:]):
            raise RuntimeError(
                f"{type(self).__name__}.compute_update()'s decay varies across this "
                f"group of {len(decays)} same-shape parameters ({decays}) -- "
                f"compute_update_batched()'s default fallback can only return one "
                f"decay shared across the whole group (see this method's own "
                f"docstring). This configuration can't use a batched-math strategy "
                f"(shape_grouped, shape_grouped_foreach) yet -- use 'simple', "
                f"'chunked', or 'foreach' instead, or give this Algorithm a real "
                f"compute_update_batched() override that handles a per-member decay."
            )
        return torch.stack(deltas, dim=0), first_decay

    @abstractmethod
    def decay_state(self, state: dict[str, Any], factor: float) -> None:
        """Scale state in place by factor. factor<=0 should behave like a
        full reset_state()."""

    @abstractmethod
    def reset_state(self, state: dict[str, Any]) -> None:
        """Reset state in place to its zero-initialized values."""
