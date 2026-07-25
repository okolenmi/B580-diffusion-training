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
        "path": Port(name="path", type=Path, required=True, path_kind="checkpoint",
                      doc="Filename under the configured checkpoints directory (subfolders allowed, "
                          "e.g. 'sdxl/base.safetensors'). Absolute paths and '..' are rejected -- this "
                          "field is reachable from the graph editor over the network, so it's sandboxed "
                          "to the configured directory regardless of what's typed here."),
    }

    def build(self, **inputs) -> dict[str, ModelWeights]:
        self.validate_inputs(inputs)
        import paths
        from safetensors.torch import load_file

        resolved = paths.resolve_safe_model_path(str(inputs["path"]), "checkpoint")
        sd = load_file(str(resolved))
        unet_sd = {k: v for k, v in sd.items() if _is_unet_key(k)}
        non_unet_sd = {k: v for k, v in sd.items() if not _is_unet_key(k)}
        result = {"weights": ModelWeights(unet_sd, non_unet_sd)}
        self.validate_outputs(result)
        return result
