"""ForeachCAMEOptimizerNode: wraps core.optimizers.ForeachXPUCAME.

Same relationship as ForeachAdafactorOptimizerNode -> ForeachXPUAdafactor:
thin pass-through, no optimizer math here. Use this instead of
CAMEOptimizerNode for LoRA -- see ForeachXPUCAME's own docstring in
core/optimizers.py for why (measured: a user's real profiler output
showed CAMEOptimizerNode's optimizer step taking ~4x the combined
forward+backward time, traced to ChunkedXPUCAME's per-parameter
device-to-host synchronization, ~3 per parameter per step).
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .handle import OptimizerHandle
from .node import OptimizerNode


class ForeachCAMEOptimizerHandle(OptimizerHandle):

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


class ForeachCAMEOptimizerNode(OptimizerNode):
    """CAME without ChunkedXPUCAME's per-parameter device syncs or its
    scratch-buffer VRAM bounding -- the right default for LoRA's small,
    fixed parameter set. See core.optimizers.ForeachXPUCAME's docstring."""

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "lr": Port(name="lr", type=float, required=False, default=1e-4,
                   doc="Learning rate (CAME's own default differs from the shared 1e-5)."),
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-30, 1e-16)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999, 0.9999)),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "verbose_profile": Port(
            name="verbose_profile", type=bool, required=False, default=False,
            doc="Per-phase timing breakdown (cast/normalize/clip/momentum+residual/update) "
                "printed every step, with a device sync between each phase for real numbers. "
                "Also prints which optimizer class is actually active either way, "
                "unconditionally, even with this off -- if you expected ForeachXPUCAME and "
                "the server log says ChunkedXPUCAME, that's the graph still wired to the old "
                "CAMEOptimizerNode, not this one.",
        ),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        from core.optimizers import ForeachXPUCAME
        legacy = ForeachXPUCAME(
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
            verbose_profile=inputs.get("verbose_profile", self.INPUTS["verbose_profile"].default),
        )
        result = {"optimizer": ForeachCAMEOptimizerHandle(legacy)}
        self.validate_outputs(result)
        return result
