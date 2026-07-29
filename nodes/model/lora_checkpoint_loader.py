"""LoRACheckpointLoaderNode: loads a saved LoRA checkpoint's weights into
a freshly-injected model, for two use cases -- continuing to train it
further, or (wired into LoRAPhaseSplitNode right after this node)
freezing it as gen0 and starting a new phase on top of a previously
trained LoRA rather than from scratch.

Both are the same operation (load weights into the current registry) --
they only differ in what's wired downstream, so one node serves both,
matching what was actually asked for rather than building two nearly-
identical nodes.

Reuses core.lora.load_lora_into_model directly (proven, and already
exercised for real in nodes/smoke_tests/smoke_test_lora_phase_split.py's
round-trip checks) -- no reimplementation. That function's own coverage
check is permissive (silently skips any registry entry whose keys
aren't in the file, rather than erroring); this node adds a stricter,
loud check in front of it -- every currently-injected layer's expected
key must be present in the checkpoint, or this raises with the specific
missing keys listed, rather than silently loading a partial LoRA. Rank
mismatches are also caught explicitly with a clear message instead of
whatever assertion core.lora.LoRALinear.load_lora_weights happens to
raise internally.

Only meaningful before any phase-split has happened (load onto a fresh
ComfyUNetLoRANode injection, plain LoRALinear/LoRAConv2d layers) --
core.lora.load_lora_into_model's own isinstance gate silently skips
LoRAGeneration layers, so loading onto an already-split model wouldn't
raise, it just wouldn't do anything to the split layers. Load first,
split after, not the other way around.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from .handle import TrainableModel
from .lora_injector import ComfyUNetTrainableModel
from .lora_phases import lora_key
from .node import LoRAInjectorNode


class LoRACheckpointLoaderNode(LoRAInjectorNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=TrainableModel, required=True,
                       doc="A freshly-injected model (e.g. straight from ComfyUNetLoRANode) "
                           "-- not one that's already been through a LoRAPhaseSplitNode."),
        "relative_path": Port(
            name="relative_path", type=str, required=True, path_kind="lora",
            doc="Path relative to the configured LoRA directory -- the same directory "
                "LoRACheckpointSaverNode writes to. Absolute paths and '..' are rejected.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)
        import paths
        from safetensors.torch import load_file

        from core.lora import LoRAConv2d, LoRALinear, load_lora_into_model

        model = inputs["model"]
        if not isinstance(model, ComfyUNetTrainableModel):
            raise TypeError(
                f"LoRACheckpointLoaderNode needs a ComfyUNetTrainableModel "
                f"(operates on its LoRA layer registry directly), got {type(model).__name__}."
            )

        resolved = paths.resolve_safe_model_path(inputs["relative_path"], "lora")
        state_dict = load_file(str(resolved))

        registry = model.raw.lora_registry
        missing_keys = []
        rank_mismatches = []
        for full_name, _parent, _attr, layer in registry:
            if not isinstance(layer, (LoRALinear, LoRAConv2d)):
                continue  # already phase-split -- see module docstring
            key = lora_key(full_name)
            down_key, up_key = f"{key}.lora_down.weight", f"{key}.lora_up.weight"
            if down_key not in state_dict or up_key not in state_dict:
                missing_keys.append(key)
                continue
            checkpoint_rank = state_dict[down_key].shape[0]
            if checkpoint_rank != layer.rank:
                rank_mismatches.append((key, checkpoint_rank, layer.rank))

        if missing_keys:
            raise ValueError(
                f"LoRACheckpointLoaderNode: {resolved} is missing {len(missing_keys)} layer(s) "
                f"this model was injected with, e.g. {missing_keys[:5]}. Was this file saved "
                f"from a model with different target_modules, or a phase-split combined "
                f"checkpoint whose rank doesn't match this injection's rank?"
            )
        if rank_mismatches:
            details = ", ".join(f"{k}: file has rank {fr}, model injected at rank {mr}"
                                 for k, fr, mr in rank_mismatches[:5])
            raise ValueError(
                f"LoRACheckpointLoaderNode: rank mismatch loading {resolved} -- {details}. "
                f"Re-inject ComfyUNetLoRANode with rank={rank_mismatches[0][1]} to match this "
                f"checkpoint (a phase-split combined checkpoint's rank is the sum of every "
                f"phase's own rank, not any single phase's)."
            )

        load_lora_into_model(registry, state_dict)

        result = {"model": model}
        self.validate_outputs(result)
        return result
