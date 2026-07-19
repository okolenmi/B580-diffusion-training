"""CAMEOptimizerNode: wraps the already-verified core.optimizers.ChunkedXPUCAME.

No optimizer math lives here -- CAMEOptimizerHandle is a thin pass-through
adapter. ChunkedXPUCAME's algorithm was numerically verified against the
official reference implementation (github.com/yangluo7/CAME) earlier this
session; that verification is unaffected by this file, since this file
never touches core/optimizers.py's code, only calls into an instance of it.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .handle import OptimizerHandle
from .node import OptimizerNode


class CAMEOptimizerHandle(OptimizerHandle):
    """Adapter: makes a core.optimizers.ChunkedXPUCAME instance satisfy the
    OptimizerHandle contract. Every method below delegates to self._legacy;
    none of them contain any numerical logic of their own."""

    def __init__(self, legacy_optimizer):
        self._legacy = legacy_optimizer

    @property
    def lr(self) -> float:
        return self._legacy.lr

    def update_lr(self, new_lr: float) -> None:
        # ChunkedXPUCAME has a plain param_lr list (no radial-multiplier
        # support wired up for it in the old optimizer_builder.py path) --
        # see handle.py's update_lr docstring for why this is explicit here
        # rather than sniffed via hasattr() by an external free function.
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


class CAMEOptimizerNode(OptimizerNode):
    """CAME (Confidence-guided Adaptive Memory Efficient Optimization),
    Luo et al. ACL 2023 -- see core.optimizers.ChunkedXPUCAME's own
    docstring for the full algorithm description and verification notes.
    """

    INPUTS: ClassVar[dict[str, Port]] = {
        **OptimizerNode.COMMON_INPUTS,
        "lr": Port(name="lr", type=float, required=False, default=1e-4,
                   doc="Learning rate (CAME's own default differs from the shared 1e-5)."),
        "eps": Port(name="eps", type=tuple, required=False, default=(1e-30, 1e-16)),
        "clip_threshold": Port(name="clip_threshold", type=float, required=False, default=1.0),
        "betas": Port(name="betas", type=tuple, required=False, default=(0.9, 0.999, 0.9999)),
        "weight_decay": Port(name="weight_decay", type=float, required=False, default=0.0),
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    def build(self, **inputs) -> dict[str, OptimizerHandle]:
        self.validate_inputs(inputs)
        from core.optimizers import ChunkedXPUCAME
        legacy = ChunkedXPUCAME(
            params=inputs["params"],
            lr=inputs.get("lr", self.INPUTS["lr"].default),
            eps=inputs.get("eps", self.INPUTS["eps"].default),
            clip_threshold=inputs.get("clip_threshold", self.INPUTS["clip_threshold"].default),
            betas=inputs.get("betas", self.INPUTS["betas"].default),
            weight_decay=inputs.get("weight_decay", self.INPUTS["weight_decay"].default),
            device=inputs.get("device", self.INPUTS["device"].default),
        )
        result = {"optimizer": CAMEOptimizerHandle(legacy)}
        self.validate_outputs(result)
        return result
