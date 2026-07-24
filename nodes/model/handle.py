"""Runtime contracts for loaded checkpoints and trainable models."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelWeights:
    """A loaded checkpoint's UNet state dict, separated from everything else
    (CLIP/VAE) in the same file. Plain data -- no behavior."""

    def __init__(self, unet_sd: dict, non_unet_sd: dict):
        self.unet_sd = unet_sd
        self.non_unet_sd = non_unet_sd


class TrainableModel(ABC):
    """Runtime contract for a model ready to be trained."""

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
