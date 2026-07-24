"""LoRACheckpointSaverNode: writes trained LoRA adapter weights to disk."""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .lora_injector import ComfyUNetTrainableModel
from .node import CheckpointSaverNode


class LoRACheckpointSaverNode(CheckpointSaverNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=ComfyUNetTrainableModel, required=True),
        "path": Port(name="path", type=str, required=True),
    }

    def build(self, **inputs) -> dict[str, str]:
        self.validate_inputs(inputs)
        from core.save import save_lora_checkpoint

        model: ComfyUNetTrainableModel = inputs["model"]
        path = str(inputs["path"])
        save_lora_checkpoint(model.raw, path)
        result = {"path": path}
        self.validate_outputs(result)
        return result
