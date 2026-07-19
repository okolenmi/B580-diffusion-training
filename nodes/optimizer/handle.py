"""OptimizerHandle: the runtime contract for a constructed optimizer.

Deliberately separate from OptimizerNode (nodes/optimizer/node.py). A Node
represents a *construction* step in the graph (config in, object out); a
Handle is the actual runtime object that construction produces, used later
during real training -- different concern, different lifetime, so it gets
its own interface rather than being folded into the Node itself. See
docs/nodes_package_design.md, "Runtime Handle ABCs".

Every method here corresponds to something this codebase's training loop
(core/train_step.py, core/trainer.py) actually calls on an optimizer today
-- this isn't a speculative interface, it's the real, exercised surface
area, made explicit and enforced instead of implicit and duck-typed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class OptimizerHandle(ABC):

    @property
    @abstractmethod
    def lr(self) -> float:
        """Current learning rate (read directly by core/train_step.py's
        progress logging). Read-only -- see update_lr() to change it."""

    @abstractmethod
    def update_lr(self, new_lr: float) -> None:
        """Change the learning rate. The old codebase did this via a free
        function (core/optimizer_builder.py's update_lr()) that branched on
        hasattr() checks -- "does this optimizer have its own update_lr
        method (FusedXPUAdafactor's per-parameter cached LR values need
        one), else does it have radial per-parameter multipliers, else does
        it have a plain param_lr list, else just set .lr directly" --
        exactly the kind of implicit, sniff-the-object interface this
        package's contracts are meant to replace. Each concrete Handle
        below implements this explicitly and correctly for its own wrapped
        legacy optimizer instead.
        """

    @abstractmethod
    def step(self, n_steps: int = 1) -> None:
        """Apply one real optimizer update. n_steps communicates how many
        micro-batches' worth of accumulated gradient this update represents
        (see core/train_step.py's grad_accum handling) -- optimizers that
        don't need this distinction are free to ignore the argument."""

    @abstractmethod
    def zero_grad(self) -> None:
        """Clear .grad on every managed parameter."""

    @abstractmethod
    def offload_states_to_cpu(self) -> None:
        """Move all optimizer state off the training device, freeing device
        memory (used between cyclic-training cache rebuilds)."""

    @abstractmethod
    def reload_states_to_device(self, device: str | None = None) -> None:
        """Move optimizer state back onto a device (None = whatever device
        this handle was originally built for)."""

    @abstractmethod
    def decay_states(self, factor: float) -> None:
        """Scale all tracked optimizer state by factor. factor<=0 should
        behave like a full reset_states()."""

    @abstractmethod
    def reset_states(self) -> None:
        """Clear all tracked optimizer state, keeping parameter references
        and step count."""

    @abstractmethod
    def free_states(self) -> None:
        """Release all optimizer state entirely (used when the optimizer
        itself is being discarded, not just paused)."""


class FusedOptimizerHandle(OptimizerHandle):
    """Extends OptimizerHandle for optimizers whose actual parameter
    updates happen inside backward-pass hooks rather than in a single,
    separate step() call.

    Found by reading core.optimizers.FusedXPUAdafactor's real
    implementation, not assumed going in: step() and zero_grad() are
    literal no-ops for this family (confirmed -- both are `pass` bodies) --
    every real update happens per-parameter, inside
    `_update_param`, triggered by a backward hook registered once at
    construction time. begin_step()/prepare_next_pass() are the real
    per-micro-step lifecycle that matters here: a caller needs to call
    begin_step() before the backward pass(es) that make up one real
    optimizer update, and prepare_next_pass() between multiple backward()
    calls accumulated into a single logical update (this codebase's
    conditional/unconditional dual-pass distillation, specifically).

    This is *not* Adafactor-specific by construction -- the contract only
    describes the fused/hook-based *execution protocol* (register once,
    begin_step before backward, prepare_next_pass between accumulated
    passes), not anything about which algorithm's math runs inside the
    hook. A future fused implementation of a different algorithm (CAME,
    say) could satisfy this same contract -- see
    docs/nodes_package_design.md's "Fused optimizer family" section for
    why that's a real possibility this interface doesn't foreclose, but
    also for why it's a substantial new algorithm-engineering task in
    core/optimizers.py, not something this adapter layer unlocks by
    itself.
    """

    @abstractmethod
    def begin_step(self, sub_steps: int = 1) -> None:
        """Reset per-update bookkeeping. sub_steps: how many physical
        backward() calls make up one real optimizer update (e.g. 2 for a
        conditional+unconditional distillation pair)."""

    @abstractmethod
    def prepare_next_pass(self) -> None:
        """Call between accumulated backward() calls within one logical
        update (when sub_steps > 1), before the next backward() call."""
