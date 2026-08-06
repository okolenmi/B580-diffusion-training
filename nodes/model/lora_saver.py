"""LoRACheckpointSaverNode: writes a trained-weights state dict to disk.

Depends on TrainedWeightsExportable (nodes/model/handle.py), not a
concrete model class. TrainableModel extends that interface, so a live,
still-training ComfyUNetTrainableModel is one implementer; a
LoRAPhaseSplitNode's completed_generation snapshot (FrozenLoRASnapshot,
nodes/model/lora_phases.py -- not a TrainableModel at all, no
forward/train/eval/to) is another. Both satisfy the same real
issubclass() check the graph editor does on declared port types, so one
saver node genuinely serves both wires rather than needing a second,
near-duplicate node class for the snapshot case.

Writes the dict straight to safetensors itself rather than going through
core.save.save_lora_checkpoint -- that legacy function reads weights via
ComfyUNetWrapper.get_lora_weights(), which core.lora's own isinstance
gate would skip for a phase-split model's LoRAGeneration layers (see
nodes/model/lora_phases.py); ComfyUNetTrainableModel.trained_state_dict()
already resolves that, and there's nothing left for the legacy function
to add on top of an already-detached, already-CPU state dict.

Known gap, not fixed here because nothing in nodes/ can trigger it yet:
core.save.save_lora_checkpoint also temporarily removes a FusedXPUAdafactor's
backward hooks during the read, to avoid a hook firing mid-save. That
race needs a *live, still-training* model and a save happening while
training is paused mid-loop -- not reachable today since no nodes/
orchestration node calls LoRACheckpointSaverNode from inside a TrainerNode's
step loop (see nodes_package_design.md's TrainerNode scope-reduction
list: "no such orchestration node exists yet"). Worth revisiting
together whenever that node gets built, not before -- and this node's
previous version, also never passed an optimizer through here either, so
this isn't a regression.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from ..components.layout import ProjectLayout
from .handle import TrainedWeightsExportable
from .node import CheckpointSaverNode


class LoRACheckpointSaverNode(CheckpointSaverNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(
            name="model", type=TrainedWeightsExportable, required=True,
            doc="A trained model (e.g. SupervisedLoRATrainerNode's output) or a "
                "LoRAPhaseSplitNode's completed_generation snapshot -- both "
                "TrainedWeightsExportable, so either wires in here.",
        ),
        "relative_path": Port(
            name="relative_path", type=str, required=True, path_kind="lora_output",
            doc="Path relative to the configured LoRA directory, e.g. "
                "'style_v2/checkpoint_1000.safetensors'. Subfolders are created as "
                "needed. Absolute paths and '..' are rejected -- this field is reachable "
                "from the graph editor over the network, so it's sandboxed to the "
                "configured directory regardless of what's typed here.",
        ),
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- see nodes/components/layout.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, str]:
        self.validate_inputs(inputs)
        from safetensors.torch import save_file

        model = inputs["model"]
        if not isinstance(model, TrainedWeightsExportable):
            raise TypeError(
                f"LoRACheckpointSaverNode needs a TrainedWeightsExportable "
                f"(a trained model or a phase-split snapshot), got {type(model).__name__}."
            )
        state_dict = model.trained_state_dict()
        if not state_dict:
            raise ValueError("LoRACheckpointSaverNode: nothing to save -- "
                              "trained_state_dict() returned an empty dict.")

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        resolved = layout.resolve_safe_model_path(inputs["relative_path"], "lora")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp = resolved.with_suffix(resolved.suffix + ".tmp")
        save_file(state_dict, str(tmp))
        tmp.replace(resolved)

        result = {"saved_path": str(resolved)}
        self.validate_outputs(result)
        return result
