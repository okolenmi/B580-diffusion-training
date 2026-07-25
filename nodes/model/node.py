"""Model domain-family ABCs: weight loading, LoRA injection, checkpoint saving."""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

from ..core import Node, Port
from .handle import ModelWeights, TrainableModel


class ModelProviderNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "weights": Port(name="weights", type=ModelWeights, required=True),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, ModelWeights]:
        ...


class LoRAInjectorNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=TrainableModel, required=True),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "weights": Port(name="weights", type=ModelWeights, required=True),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, TrainableModel]:
        ...


class CheckpointSaverNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "saved_path": Port(name="saved_path", type=str, required=True,
                            doc="The resolved absolute path the checkpoint was actually written to."),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=TrainableModel, required=True),
        "relative_path": Port(name="relative_path", type=str, required=True,
                               doc="Path relative to the configured directory for this checkpoint kind."),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, str]:
        ...
