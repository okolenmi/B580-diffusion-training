"""SafetensorsCheckpointNode: loads a checkpoint, splits UNet from the rest."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..core import Port
from .handle import ModelWeights
from .node import ModelProviderNode


def _is_unet_key(key: str) -> bool:
    return key.startswith("model.diffusion_model.")


class SafetensorsCheckpointNode(ModelProviderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "path": Port(name="path", type=Path, required=True),
    }

    def build(self, **inputs) -> dict[str, ModelWeights]:
        self.validate_inputs(inputs)
        from safetensors.torch import load_file

        sd = load_file(str(inputs["path"]))
        unet_sd = {k: v for k, v in sd.items() if _is_unet_key(k)}
        non_unet_sd = {k: v for k, v in sd.items() if not _is_unet_key(k)}
        result = {"weights": ModelWeights(unet_sd, non_unet_sd)}
        self.validate_outputs(result)
        return result
