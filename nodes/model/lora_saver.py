"""LoRACheckpointSaverNode: writes trained LoRA adapter weights to disk."""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .lora_injector import ComfyUNetTrainableModel
from .node import CheckpointSaverNode


class LoRACheckpointSaverNode(CheckpointSaverNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=ComfyUNetTrainableModel, required=True),
        "relative_path": Port(
            name="relative_path", type=str, required=True, path_kind="lora_output",
            doc="Path relative to the configured LoRA directory, e.g. "
                "'style_v2/checkpoint_1000.safetensors'. Subfolders are created as "
                "needed. Absolute paths and '..' are rejected -- this field is reachable "
                "from the graph editor over the network, so it's sandboxed to the "
                "configured directory regardless of what's typed here.",
        ),
    }

    def build(self, **inputs) -> dict[str, str]:
        self.validate_inputs(inputs)
        import paths
        from core.save import save_lora_checkpoint

        model: ComfyUNetTrainableModel = inputs["model"]
        resolved = paths.resolve_safe_model_path(inputs["relative_path"], "lora")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        save_lora_checkpoint(model.raw, str(resolved))
        result = {"saved_path": str(resolved)}
        self.validate_outputs(result)
        return result
