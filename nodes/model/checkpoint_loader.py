"""SafetensorsCheckpointNode: loads a checkpoint, splits UNet from the rest."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..core import Port
from ..components.layout import ProjectLayout
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
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- today's exact directory resolution "
                "(paths.py's set_comfy_dir()/environment/.env), just snapshotted once instead of "
                "read from module-global state at call time. See nodes/components/layout.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, ModelWeights]:
        self.validate_inputs(inputs)
        from safetensors.torch import load_file

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        resolved = layout.resolve_safe_model_path(str(inputs["path"]), "checkpoint")
        sd = load_file(str(resolved))
        unet_sd = {k: v for k, v in sd.items() if _is_unet_key(k)}
        non_unet_sd = {k: v for k, v in sd.items() if not _is_unet_key(k)}
        result = {"weights": ModelWeights(unet_sd, non_unet_sd)}
        self.validate_outputs(result)
        return result
