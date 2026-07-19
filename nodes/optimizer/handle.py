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
