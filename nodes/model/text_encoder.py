"""TextEncoderNode: builds a text encoder from checkpoint weights (SDXL dual CLIP)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..core import Node, Port
from .handle import ModelWeights


class TextEncoder(ABC):

    @abstractmethod
    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        """Return (context, pooled_y) tensors for the UNet's conditioning inputs."""

    @abstractmethod
    def unload(self) -> None:
        ...


class TextEncoderNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "encoder": Port(name="encoder", type=TextEncoder, required=True),
    }

    COMMON_INPUTS: ClassVar[dict[str, Port]] = {
        "weights": Port(name="weights", type=ModelWeights, required=True),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, TextEncoder]:
        ...


class SDXLTextEncoder(TextEncoder):

    def __init__(self, legacy_encoder):
        self._legacy = legacy_encoder

    def encode(self, prompt: str, batch_size: int, height: int, width: int):
        return self._legacy.encode_for_unet(prompt, batch_size=batch_size, height=height, width=width)

    def unload(self) -> None:
        self._legacy.unload()


class SDXLTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **TextEncoderNode.COMMON_INPUTS,
        "device": Port(name="device", type=str, required=False, default="xpu"),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        from core.clip_encode import SDXLClipEncoder

        weights: ModelWeights = inputs["weights"]
        legacy = SDXLClipEncoder(weights.non_unet_sd,
                                  device=inputs.get("device", self.INPUTS["device"].default))
        result = {"encoder": SDXLTextEncoder(legacy)}
        self.validate_outputs(result)
        return result
