"""Runtime contracts for loaded checkpoints and trainable models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelWeights:
    """A loaded checkpoint's UNet state dict, separated from everything else
    (CLIP/VAE) in the same file. Plain data -- no behavior."""

    def __init__(self, unet_sd: dict, non_unet_sd: dict):
        self.unet_sd = unet_sd
        self.non_unet_sd = non_unet_sd


class ParameterList(list):
    """A list of trainable tensors, typed distinctly from a plain `Any` so
    an OptimizerNode's `params` input can reject an accidental TrainableModel
    connection outright instead of accepting it (Any accepts everything)
    and crashing later inside the real optimizer constructor. Genuinely
    just a list otherwise -- torch optimizers accept any iterable, so this
    is a free drop-in, not a wrapper callers need to unwrap."""


class TrainedWeightsExportable(ABC):
    """Anything that can hand back a plain, save-ready adapter state dict
    (CPU tensors, keyed for safetensors). What LoRACheckpointSaverNode
    actually depends on. TrainableModel extends this below rather than
    the saver node depending on TrainableModel directly, specifically so
    a second, non-trainable implementer -- a frozen, already-extracted
    snapshot with no forward/train/eval/to (see FrozenLoRASnapshot in
    nodes/model/lora_phases.py) -- is a normal, separately-typed ABC
    implementer the graph's own issubclass() port check accepts at the
    same saver input, not a special case the saver has to know about."""

    @abstractmethod
    def trained_state_dict(self) -> dict:
        ...


class TrainableModel(TrainedWeightsExportable, ABC):
    """Runtime contract for a model ready to be trained. Extending
    TrainedWeightsExportable means every TrainableModel is required to
    say how to export what it learned -- a reasonable universal
    requirement (not a LoRA-specific one; a hypothetical future
    full-finetune TrainableModel would just export its own full weight
    diff through the same method), and the concrete reason
    LoRACheckpointSaverNode's model input can stay typed at this ABC
    without narrowing to a concrete class."""

    @abstractmethod
    def forward(self, x_t, timestep, context, y):
        ...

    @abstractmethod
    def trainable_parameters(self) -> list:
        ...

    @abstractmethod
    def train(self) -> "TrainableModel":
        ...

    @abstractmethod
    def eval(self) -> "TrainableModel":
        ...

    @abstractmethod
    def to(self, device=None, **kwargs) -> "TrainableModel":
        ...
