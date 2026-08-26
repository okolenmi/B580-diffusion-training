"""SafetensorsCheckpointNode: resolves and validates a checkpoint path,
splits UNet from the rest. Lazy since Phase 1 of the resources-
controller redesign (docs/resources_controller_redesign_plan.md) --
this no longer loads the file itself; ModelWeights (nodes/model/
handle.py) does that on first real access, or not at all if nothing
downstream ever touches unet_sd/non_unet_sd. build() still does real,
useful work up front: resolving+sandboxing the path (unchanged from
before) and a cheap existence/format check (safe_open() opening the
file and reading its header) that turns a bad path into a clear error
right here, at graph-construction time, rather than a confusing one
buried inside whatever node first happens to trigger the real load.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from ..core import Port
from ..components.layout import ProjectLayout
from .handle import ModelWeights
from .node import ModelProviderNode


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
        from safetensors import safe_open

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        resolved = layout.resolve_safe_model_path(str(inputs["path"]), "checkpoint")
        # Cheap (header-only, see resource_inspection.py's own docstring
        # for why this doesn't touch tensor data) existence/format check
        # -- fails loudly here, not the first time something downstream
        # happens to access .unet_sd and triggers the real load.
        try:
            with safe_open(str(resolved), framework="pt") as f:
                next(iter(f.keys()), None)
        except Exception as e:
            raise ValueError(
                f"SafetensorsCheckpointNode: {resolved} doesn't look like a valid "
                f"safetensors file ({e})."
            ) from e
        result = {"weights": ModelWeights(resolved)}
        self.validate_outputs(result)
        return result
