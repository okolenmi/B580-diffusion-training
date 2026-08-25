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
round-trip checks) -- no reimplementation, for the plain-LoRA layers it
actually understands. That function's own coverage check is permissive
(silently skips any registry entry whose keys aren't in the file, rather
than erroring); this node adds a stricter, loud check in front of it --
every currently-injected layer's expected key must be present in the
checkpoint, or this raises with the specific missing keys listed, rather
than silently loading a partial LoRA. Rank mismatches are also caught
explicitly with a clear message instead of whatever assertion
core.lora.LoRALinear.load_lora_weights happens to raise internally.

A real gap this same "loud, not silent" discipline used to miss
entirely, found and closed while wiring in DoRA's `.dora_scale` (design
doc section 3.1/9.1): core.lora.load_lora_into_model's isinstance gate
is `(LoRALinear, LoRAConv2d)` -- exactly right for skipping an
already-phase-split LoRAGeneration layer (this module's own original
docstring note), but DoRALinear/DoRAConv2d (nodes/model/dora_layer.py)
are ALSO not instances of those two classes (built via composition over
one, not inheritance -- see dora_layer.py's own docstring for why). So
the legacy call silently did nothing at all for a DoRA-adapted layer --
not just "misses magnitude," the direction (A/B) and alpha were never
restored either, with no error and nothing printed. core/lora.py is
frozen (this project's standing rule), so the fix lives entirely here:
_load_dora_layers() below handles DoRA layers itself, using the same
loud-missing-keys/rank-mismatch discipline as the plain-LoRA path above
it, and the same alpha-restore-with-a-printed-note behavior
load_lora_into_model already gives plain layers -- via
DoRALinear/DoRAConv2d's own new restore_alpha() method (dora_layer.py),
not by this module reaching into their internals directly: those
classes keep their own alpha/scaling copy (forward() reads
self.scaling, not self._lora.scaling), so recomputing the restored
value correctly is their job, the same way load_lora_weights()/
load_dora_weights() already are.

Only meaningful before any phase-split has happened (load onto a fresh
ComfyUNetLoRANode injection, plain LoRALinear/LoRAConv2d or
DoRALinear/DoRAConv2d layers) -- core.lora.load_lora_into_model's own
isinstance gate silently skips LoRAGeneration layers, and
_load_dora_layers() below deliberately matches that same "skip, don't
error" choice for a DoRA layer that's already been phase-split (its
`inner` holds the trained DoRA weights this checkpoint would restore
into, but `inner` isn't reachable through the registry's current
top-of-stack entries at all -- there's no layer object here to load
onto, the same reason a plain already-split layer is skipped rather than
erroring). Load first, split after, not the other way around.
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
    """The DoRA-layer half of LoRACheckpointLoaderNode.build() --
    core.lora.load_lora_into_model (frozen legacy code) silently skips
    every DoRALinear/DoRAConv2d entirely, see this module's own
    docstring for exactly why. Missing-keys/rank-mismatch validation for
    these layers already happened in build() before this is called (same
    loud discipline as the plain-LoRA path, just gathered together
    there); this only does the actual loading, once validation has
    already passed.

    Magnitude (`.dora_scale`) is the one piece allowed to be genuinely
    optional here, not just "validated already": a checkpoint saved
    before this project's DoRA round-trip existed (or saved by
    something else that only ever wrote the standard three keys) has
    direction only, and DoRALinear.load_lora_weights()'s own documented
    behavior -- recompute magnitude fresh from the loaded direction --
    is a real, legitimate thing to load in that case (dora_layer.py's
    module docstring: "start DoRA training from an existing plain-LoRA
    checkpoint's direction"), not a degraded fallback to apologize for.
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
            print(f"    [DoRA] {full_name}: no {magnitude_key!r} in checkpoint -- "
                  f"loading direction only, magnitude recomputed fresh from it "
                  f"(see DoRALinear.load_lora_weights' docstring).")
            layer.load_lora_weights(A, B)

        alpha_key = f"{key}.alpha"
        if alpha_key in state_dict:
            saved_alpha = state_dict[alpha_key].item()
            if abs(saved_alpha - layer.alpha) > 1e-6:
                print(f"    [DoRA] {full_name}: alpha mismatch "
                      f"(checkpoint={saved_alpha}, config={layer.alpha}). "
                      f"Using checkpoint value.")
            layer.restore_alpha(saved_alpha)


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

        from core.lora import LoRAConv2d, LoRALinear, load_lora_into_model

        from .dora_layer import DoRAConv2d, DoRALinear

        model = inputs["model"]
        if not isinstance(model, ComfyUNetTrainableModel):
            raise TypeError(
                f"LoRACheckpointLoaderNode needs a ComfyUNetTrainableModel "
                f"(operates on its LoRA layer registry directly), got {type(model).__name__}."
            )

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        resolved = layout.resolve_safe_model_path(inputs["relative_path"], "lora")
        state_dict = load_file(str(resolved))

        registry = model.raw.lora_registry
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

        load_lora_into_model(registry, state_dict)  # plain LoRALinear/LoRAConv2d layers
        _load_dora_layers(registry, state_dict)      # DoRALinear/DoRAConv2d layers

        result = {"model": model}
        self.validate_outputs(result)
        return result
