"""AdafactorOptimizerNode: wraps core.optimizers.ChunkedXPUAdafactor.

Thin pass-through adapter, no optimizer math reimplemented -- see
came.py's module docstring for the same note, which applies equally here.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .handle import OptimizerHandle
from .node import OptimizerNode


class AdafactorOptimizerHandle(OptimizerHandle):

    def __init__(self, legacy_optimizer):
        self._legacy = legacy_optimizer

    @property
    def lr(self) -> float:
        return self._legacy.lr

    def update_lr(self, new_lr: float) -> None:
        self._legacy.lr = new_lr
        self._legacy.param_lr = [new_lr] * len(self._legacy.params)

    def step(self, n_steps: int = 1) -> None:
        self._legacy.step(n_steps=n_steps)

    def zero_grad(self) -> None:
        self._legacy.zero_grad()

    def offload_states_to_cpu(self) -> None:
        self._legacy.offload_states_to_cpu()

    def reload_states_to_device(self, device: str | None = None) -> None:
        self._legacy.reload_states_to_device(device)

    def decay_states(self, factor: float) -> None:
        self._legacy.decay_states(factor)

    def reset_states(self) -> None:
        self._legacy.reset_states()

    def free_states(self) -> None:
        self._legacy.free_states()


class AdafactorOptimizerNode(OptimizerNode):
    """Chunked GPU Adafactor with memory-pool & scratch buffer -- see
    core.optimizers.ChunkedXPUAdafactor's own module comment for the full
    memory-management design."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-08, 1e-3)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "beta1": Port(name="beta1", type=float, required=False, default=None,
                     doc="None = Adafactor's own time-varying rho_t schedule."),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=1.0),
        "scale_parameter": Port(name="scale_parameter", type=bool, required=False, default=True),
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        from core.optimizers import ChunkedXPUAdafactor
        legacy = ChunkedXPUAdafactor(
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            beta1=inputs.get("beta1", self.INPUTS["beta1"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
            scale_parameter=inputs.get("scale_parameter", self.INPUTS["scale_parameter"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
        )
        result = {"optimizer": AdafactorOptimizerHandle(legacy)}
        self.validate_outputs(result)
        return result
