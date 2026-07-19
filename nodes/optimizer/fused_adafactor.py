"""FusedAdafactorOptimizerNode: wraps core.optimizers.FusedXPUAdafactor.

Satisfies FusedOptimizerHandle, not the plain OptimizerHandle -- see
handle.py's FusedOptimizerHandle docstring for why this family needs a
genuinely extended contract rather than being force-fit into the same
interface as the other four optimizers.

build() calls the wrapped instance's register_hooks() as its last step.
In the old code this is a required, easy-to-forget manual call site
(core/trainer.py calls it once, right after construction, gated on an
isinstance check) -- doing it automatically here means the produced Handle
is *always* immediately functional the moment build() returns, matching
every other adapter in this package, rather than leaking a
class-specific setup quirk to whoever consumes this node.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .handle import FusedOptimizerHandle
from .node import OptimizerNode


class FusedAdafactorOptimizerHandle(FusedOptimizerHandle):

    def __init__(self, legacy_optimizer):
        self._legacy = legacy_optimizer

    @property
    def lr(self) -> float:
        return self._legacy.lr

    def update_lr(self, new_lr: float) -> None:
        # Unlike the other four adapters, FusedXPUAdafactor already has its
        # own correct update_lr() (it needs to also resync _param_lr_f, the
        # plain-float cache read inside the backward hook's hot path) --
        # verified by reading it directly, so this can delegate rather than
        # reimplement the logic.
        self._legacy.update_lr(new_lr)

    def step(self, n_steps: int = 1) -> None:
        # Real no-op on the legacy class -- see FusedOptimizerHandle's
        # docstring for why. Delegating anyway (rather than a bare `pass`
        # here) keeps this adapter correct even if a future change to
        # FusedXPUAdafactor.step() ever makes it do something.
        self._legacy.step()

    def zero_grad(self) -> None:
        self._legacy.zero_grad()  # also a real no-op on the legacy class

    def begin_step(self, sub_steps: int = 1) -> None:
        self._legacy.begin_step(sub_steps=sub_steps)

    def prepare_next_pass(self) -> None:
        self._legacy.prepare_next_pass()

    def offload_states_to_cpu(self) -> None:
        self._legacy.offload_states_to_cpu()

    def reload_states_to_device(self, device: str | None = None) -> None:
        self._legacy.reload_states_to_device(device)

    def decay_states(self, factor: float) -> None:
        self._legacy.decay_states(factor)

    def reset_states(self) -> None:
        self._legacy.reset_states()

    def free_states(self) -> None:
        # Legacy free_states() also calls remove_hooks() internally --
        # confirmed by reading it -- so this correctly tears down the
        # backward hooks registered by build(), not just the state tensors.
        self._legacy.free_states()


class FusedAdafactorOptimizerNode(OptimizerNode):
    """Adafactor fused into backward-pass hooks -- per-parameter updates
    happen as each parameter's gradient becomes available during
    backward(), rather than in a separate step() call. See
    core.optimizers.FusedXPUAdafactor's own docstring/comments and
    handle.py's FusedOptimizerHandle for the execution-model details.
    """

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-08, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=1.0),
        "scale_parameter": Port(name="scale_parameter", type=bool, required=False, default=True),
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }
    # Overridden so the graph/introspection layer can see this node produces
    # a FusedOptimizerHandle specifically -- a real subtype of
    # OptimizerHandle, not just "optimizer" generically. A consumer that
    # needs the extra begin_step/prepare_next_pass methods (i.e. the actual
    # training loop, once nodes/ is wired into it -- a later phase) can use
    # this type information instead of an isinstance check against a
    # concrete legacy class, which is what core/trainer.py does today.
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "optimizer": Port(
            name="optimizer", type=FusedOptimizerHandle, required=True,
            doc="A constructed, ready-to-use fused (backward-hook-based) optimizer.",
        ),
    }

    def build(self, **inputs) -> dict[str, FusedOptimizerHandle]:
        self.validate_inputs(inputs)
        from core.optimizers import FusedXPUAdafactor
        legacy = FusedXPUAdafactor(
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            beta1=inputs.get("beta1", self.INPUTS["beta1"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
            scale_parameter=inputs.get("scale_parameter", self.INPUTS["scale_parameter"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
        )
        legacy.register_hooks()
        result = {"optimizer": FusedAdafactorOptimizerHandle(legacy)}
        self.validate_outputs(result)
        return result
