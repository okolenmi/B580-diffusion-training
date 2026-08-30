"""load_lora_into_registry(): loads a saved LoRA's weights into every
matching layer of a LoRA registry (a freshly-injected model's
wrapper.lora_registry). LoRACheckpointLoaderNode wraps this for the
node-graph case (load from a file path); LoRATrainingSkeleton's own
continue_lora_sd parameter (lora_training_resources.py) calls it
directly with an already-in-memory state dict -- same validation and
loading either way, not two copies of it.

Reuses core.lora.load_lora_into_model for the plain-LoRA layers it
understands. That function's own coverage check is permissive --
silently skips any registry entry whose keys aren't in the source,
rather than erroring. load_lora_into_registry adds a stricter check in
front of it: every currently-injected layer's expected key must be
present, or this raises with the specific missing keys listed, rather
than silently loading a partial LoRA. Rank mismatches are also caught
explicitly with a clear message.

core.lora.load_lora_into_model's isinstance gate is
(LoRALinear, LoRAConv2d) -- correct for skipping an already-phase-split
LoRAGeneration layer, but DoRALinear/DoRAConv2d (dora_layer.py) are
also not instances of those two classes (composition, not inheritance),
so that call silently does nothing for a DoRA-adapted layer: direction
and alpha are never restored, not just magnitude. core/lora.py is
frozen, so _load_dora_layers() below handles DoRA layers itself, same
missing-keys/rank-mismatch discipline as the plain-LoRA path, and the
same alpha-restore-with-a-printed-note behavior load_lora_into_model
gives plain layers, via DoRALinear/DoRAConv2d's own restore_alpha()
method.

Only meaningful before any phase-split has happened. load_lora_into_model's
own isinstance gate silently skips LoRAGeneration layers, and
_load_dora_layers() matches that same skip for an already-phase-split
DoRA layer (its `inner` holds the weights this would restore into, but
isn't reachable through the registry's current top-of-stack entries).
Load first, split after.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from ..components.layout import ProjectLayout
from .handle import TrainableModel
from .lora_injector import ComfyUNetTrainableModel
from .lora_phases import lora_key
from .node import LoRAInjectorNode


def _load_dora_layers(registry, state_dict: dict) -> None:
    """The DoRA-layer half of load_lora_into_registry() --
    core.lora.load_lora_into_model silently skips every DoRALinear/
    DoRAConv2d, see this module's own docstring. Missing-keys/rank-
    mismatch validation for these layers already happened before this
    is called; this only does the actual loading.

    Magnitude (.dora_scale) is optional: a source saved before this
    project's DoRA round-trip existed has direction only, and
    DoRALinear.load_lora_weights()'s own documented behavior --
    recompute magnitude fresh from the loaded direction -- is a
    legitimate thing to do in that case (start DoRA training from an
    existing plain-LoRA source's direction), not a degraded fallback.
    Printed either way so which path ran is never silent.
    """
    from .dora_layer import DoRAConv2d, DoRALinear

    for full_name, _parent, _attr, layer in registry:
        if not isinstance(layer, (DoRALinear, DoRAConv2d)):
            continue
        key = lora_key(full_name)
        down_key, up_key = f"{key}.lora_down.weight", f"{key}.lora_up.weight"
        magnitude_key = f"{key}.dora_scale"
        A, B = state_dict[down_key], state_dict[up_key]
        if magnitude_key in state_dict:
            layer.load_dora_weights(A, B, state_dict[magnitude_key])
        else:
            print(f"    [DoRA] {full_name}: no {magnitude_key!r} in source -- "
                  f"loading direction only, magnitude recomputed fresh from it "
                  f"(see DoRALinear.load_lora_weights' docstring).")
            layer.load_lora_weights(A, B)

        alpha_key = f"{key}.alpha"
        if alpha_key in state_dict:
            saved_alpha = state_dict[alpha_key].item()
            if abs(saved_alpha - layer.alpha) > 1e-6:
                print(f"    [DoRA] {full_name}: alpha mismatch "
                      f"(source={saved_alpha}, config={layer.alpha}). "
                      f"Using source value.")
            layer.restore_alpha(saved_alpha)


def load_lora_into_registry(registry, state_dict: dict, source_description: str = "source") -> None:
    """Validates state_dict against every layer in registry (missing
    keys, rank mismatches -- raises ValueError with specifics rather
    than silently loading a partial LoRA), then loads it: plain layers
    via core.lora.load_lora_into_model, DoRA layers via
    _load_dora_layers() above. source_description is used only in error
    messages -- a file path, or a plain label like "continue_lora_sd"
    when there's no path (an already-in-memory state dict)."""
    from core.lora import LoRAConv2d, LoRALinear, load_lora_into_model

    from .dora_layer import DoRAConv2d, DoRALinear

    missing_keys = []
    rank_mismatches = []
    for full_name, _parent, _attr, layer in registry:
        if not isinstance(layer, (LoRALinear, LoRAConv2d, DoRALinear, DoRAConv2d)):
            continue  # already phase-split -- see module docstring
        key = lora_key(full_name)
        down_key, up_key = f"{key}.lora_down.weight", f"{key}.lora_up.weight"
        if down_key not in state_dict or up_key not in state_dict:
            missing_keys.append(key)
            continue
        source_rank = state_dict[down_key].shape[0]
        if source_rank != layer.rank:
            rank_mismatches.append((key, source_rank, layer.rank))

    if missing_keys:
        raise ValueError(
            f"load_lora_into_registry: {source_description} is missing {len(missing_keys)} "
            f"layer(s) this model was injected with, e.g. {missing_keys[:5]}. Was this saved "
            f"from a model with different target_modules, or a phase-split combined "
            f"checkpoint whose rank doesn't match this injection's rank?"
        )
    if rank_mismatches:
        details = ", ".join(f"{k}: source has rank {sr}, model injected at rank {mr}"
                             for k, sr, mr in rank_mismatches[:5])
        raise ValueError(
            f"load_lora_into_registry: rank mismatch loading {source_description} -- {details}. "
            f"Re-inject at rank={rank_mismatches[0][1]} to match (a phase-split combined "
            f"checkpoint's rank is the sum of every phase's own rank, not any single phase's)."
        )

    load_lora_into_model(registry, state_dict)  # plain LoRALinear/LoRAConv2d layers
    _load_dora_layers(registry, state_dict)      # DoRALinear/DoRAConv2d layers


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
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- see nodes/components/layout.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)
        from safetensors.torch import load_file

        model = inputs["model"]
        if not isinstance(model, ComfyUNetTrainableModel):
            raise TypeError(
                f"LoRACheckpointLoaderNode needs a ComfyUNetTrainableModel "
                f"(operates on its LoRA layer registry directly), got {type(model).__name__}."
            )

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        resolved = layout.resolve_safe_model_path(inputs["relative_path"], "lora")
        state_dict = load_file(str(resolved))

        load_lora_into_registry(model.raw.lora_registry, state_dict, source_description=str(resolved))

        result = {"model": model}
        self.validate_outputs(result)
        return result
