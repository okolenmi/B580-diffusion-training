"""ComposedFusedOptimizerHandle: any Algorithm, executed via backward
hooks instead of an explicit step() call.

Why this isn't an ExecutionStrategy, unlike everything else in
strategies/: an ExecutionStrategy's step() is called explicitly, after
backward() has already finished, and loops over every parameter. Fused
execution's entire point is the opposite -- each parameter updates itself
the moment *its own* gradient is ready, from inside backward(), before
backward() has even returned. Forcing that into "a strategy whose step()
ComposedOptimizerHandle.step() calls" doesn't fit: step() would never
actually run (see below), so there's no meaningful step() to call it
from. This is a different Handle-level composition instead, not a
different strategy plugged into the same one.

What *is* reused, not reinvented: subclasses ComposedOptimizerHandle
directly, so offload_states_to_cpu/reload_states_to_device/decay_states/
reset_states -- all real, all generic over "a list of per-parameter state
dicts", written once there -- work here completely unmodified. The
per-parameter math is the same algorithm.compute_update() call and the
same strategies.base.apply_update() every non-batched execution path
uses. What's actually new here is small and specific to the hook-driven
execution model: hook registration/teardown, and the multi-pass
("sub_steps") accumulation state machine below.

Confirmed algorithm-agnostic by construction (not just asserted): the
same class drives CAMEAlgorithm, AdafactorAlgorithm, and AdamWAlgorithm
in composed_fused_came.py/composed_fused_adafactor.py/composed_fused_adamw.py,
no algorithm-specific code anywhere in this file. See
docs/nodes_package_design.md's "Fifth data point" section for the
equivalence writeup (against core.optimizers.FusedXPUAdafactor, the one
legacy fused class that exists) and one deliberate, documented divergence
from it: FusedXPUAdafactor's TINY_NUMEL special case (a real formula
change for small parameters -- full elementwise second-moment tracking
instead of the row/col factored approximation, not merely a storage
optimization, confirmed by reading it directly) is Adafactor-specific
algorithm work, not a fused-execution concern, and isn't replicated here
-- this handle always uses whatever formula the Algorithm it's given
computes, same as every other composed node.

**The multi-pass state machine, freshly derived, not copied** (matched to
FusedXPUAdafactor's real behavior by reading it, then re-derived from
what the *contract* actually requires -- see the simplification note on
`algorithm.begin_step()` below): `begin_step(sub_steps)` starts a new
logical optimizer step that will span exactly `sub_steps` backward()
calls (>1 for this codebase's conditional+unconditional distillation
pair). Gradients accumulate via autograd's own default add-into-`.grad`
behavior across those calls -- nothing here needs to do anything for that
part. Each parameter's hook fires once per backward() pass it's part of;
only on the *last* pass (`_current_sub_step >= sub_steps_required`) does
it actually read `.grad` and apply an update -- earlier passes return
immediately, leaving `.grad` untouched for autograd to keep accumulating
into. `prepare_next_pass()` must be called between each backward() call
within one logical step so the next pass's first-hook-firing is detected
correctly (mirrors the legacy contract exactly, confirmed against
FusedOptimizerHandle's own docstring).

**One real simplification found while re-deriving this, not carried over
from the legacy version:** FusedXPUAdafactor advances its own `t`/`rho_t`
lazily, inside the hook, on the first hook-firing of the last pass --
because at construction/begin_step() time it doesn't yet know which
parameter's hook will fire first. That workaround isn't needed here:
`Algorithm.begin_step()` is a lifecycle hook already designed to be
called exactly once per logical step by whatever's driving it (see
algorithms/base.py) -- and `begin_step(sub_steps)` below *is* called
exactly once per logical step, by contract, before any of that step's
backward() calls happen. So it just calls `self.algorithm.begin_step(1)`
directly, synchronously, right there -- simpler, and provably equivalent
(same call count, same timing relative to the parameters' own updates)
without needing to track "was this the first hook firing" for that
specific purpose.
"""

from __future__ import annotations

from .algorithms.base import Algorithm
from .composed import ComposedOptimizerHandle, ParameterGroupPolicy
from .handle import FusedOptimizerHandle
from .strategies.base import ExecutionStrategy, apply_update


class _NoStepStrategy(ExecutionStrategy):
    """Placeholder passed to ComposedOptimizerHandle.__init__ purely so
    this class can inherit its offload/reload/decay/reset/free_states
    implementations for free (see module docstring) -- step()/zero_grad()
    on *this* object are never actually called: ComposedFusedOptimizerHandle
    overrides both to no-ops directly, matching FusedOptimizerHandle's
    documented contract (real updates happen in the backward hook, not in
    step()). offload_extra/reload_extra/free_extra keep ExecutionStrategy's
    own no-op defaults, correctly -- this placeholder has no extra state of
    its own to move or free."""

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        raise AssertionError("_NoStepStrategy.step() should never be called -- "
                              "ComposedFusedOptimizerHandle.step() is a no-op "
                              "and never delegates to it")

    def zero_grad(self, params) -> None:
        raise AssertionError("_NoStepStrategy.zero_grad() should never be called -- "
                              "ComposedFusedOptimizerHandle.zero_grad() is a no-op "
                              "and never delegates to it")


class ComposedFusedOptimizerHandle(ComposedOptimizerHandle, FusedOptimizerHandle):

    def __init__(self, algorithm: Algorithm, params, lr: float, device,
                 group_policy: ParameterGroupPolicy | None = None):
        super().__init__(algorithm=algorithm, strategy=_NoStepStrategy(),
                          params=params, lr=lr, device=device, group_policy=group_policy)
        self._hooks = []
        self._in_backward = False
        self._sub_steps_required = 1
        self._current_sub_step = 0
        self._register_hooks()

    def _register_hooks(self) -> None:
        for i, p in enumerate(self.params):
            if p.requires_grad:
                self._hooks.append(p.register_post_accumulate_grad_hook(self._on_grad_ready(i)))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _on_grad_ready(self, i: int):
        def hook(p) -> None:
            if p.grad is None:
                return
            if not self._in_backward:
                self._in_backward = True
                self._current_sub_step += 1
            if self._current_sub_step < self._sub_steps_required:
                return
            grad = p.grad.detach().float()
            p.grad = None
            delta, decay = self.algorithm.compute_update(grad, p, self.states[i], self.param_lr[i])
            apply_update(p, delta, decay)
        return hook

    def begin_step(self, sub_steps: int = 1) -> None:
        self.algorithm.begin_step(1)
        self._in_backward = False
        self._sub_steps_required = sub_steps
        self._current_sub_step = 0

    def prepare_next_pass(self) -> None:
        self._in_backward = False

    def step(self, n_steps: int = 1) -> None:
        pass  # Real updates happen in the hook -- see FusedOptimizerHandle's docstring.

    def zero_grad(self) -> None:
        pass  # Same -- the hook clears p.grad itself once it's consumed.

    def free_states(self) -> None:
        self._remove_hooks()
        super().free_states()
