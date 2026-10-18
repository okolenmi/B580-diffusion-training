"""Values that travel between training nodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .optim.handle import ParamGroup
from .runtime.residents import DeviceResident, module_bytes


@dataclass
class TrainableDenoiser(DeviceResident):
    """A denoiser with adapters injected: what trains, what saves, what the trainer's forward runs."""

    family: Any
    part: Any  # the DenoiserPart it was loaded from
    denoiser: Any
    adapted: Any  # AdaptedPart
    device: str
    rank: int
    alpha: float
    kind: str
    selection: Any = None
    gate_window: Any = None
    info: dict = field(default_factory=dict)
    checkpointer: Any = None  # what recomputes activations in backward; None when checkpointing is off
    checkpoint_share: float = 1.0  # share of the transformer blocks that use it
    deterministic: bool = False  # no layer draws random numbers in the forward (dropout off), so a step can be captured

    def set_checkpoint_share(self, share: float) -> None:
        if self.checkpointer is None:
            raise ValueError("this denoiser was built without checkpointing")
        self.denoiser.set_checkpointer(self.checkpointer, share)
        self.checkpoint_share = share

    @property
    def gate(self):
        return self.adapted.gate

    def param_groups(self) -> list[ParamGroup]:
        return [ParamGroup(g["params"], g["lr_mult"], g["name"]) for g in self.adapted.param_groups()]

    def footprint_bytes(self) -> int:
        return module_bytes(self.denoiser.module())

    def offload(self) -> None:
        self.denoiser.to("cpu")

    def reload(self, device: str | None = None) -> None:
        self.denoiser.to(device or self.device)

    def release(self) -> None:
        self.denoiser = self.adapted = None
