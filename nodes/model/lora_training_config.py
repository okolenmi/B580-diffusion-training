"""LoRATrainingConfigNode: docs/resources_controller_redesign_plan.md's
Phase 6. Takes Phase 5's `LoRATrainingResources` (verified, NOT yet
LoRA-injected) and actually creates the trainable adapter: rank, alpha,
frozen-weight-storage -- the things Phase 5 deliberately excluded, see
that module's own docstring for why (rank/alpha/storage are properties
of an *injection*, not of a verified *resource*).

Dispatches on `resources`'s own concrete type to find the matching
`LoRATrainingSkeleton` subclass to build (`_TRAINER_FOR_RESOURCES`
below) -- there's exactly one (task, architecture) pair today
(`SDXL_LoRATrainingResources` -> `SDXL_LoraTrainer`), the same one
Phase 5 itself has today. No separate "preset" selector here, unlike
`ResourcesControllerNode`: whatever's wired into `resources` already
decided the architecture upstream, this node only needs to know which
trainer class matches it. Grows the same way `_PRESETS` does in
`resources_controller.py` -- one more dict entry per architecture, not
a rewrite.

The one real, concrete job Phase 5 explicitly deferred to this node:
locking rank when continuing training from an existing LoRA.
`resources.continue_lora_sd`'s own shape (`lora_down.weight`'s own
first dimension) already *is* the rank -- nothing anywhere resizes it
-- so the `rank` input is a free choice only when there's no
`continue_lora_sd` to match; otherwise it's ignored entirely and the
continuing LoRA's own detected rank is used instead, matching direct
feedback that this should be "impossible to override," not a value
someone could accidentally mismatch and only find out from a shape-
mismatch error deep inside torch the first time the continuing LoRA's
own tensors get loaded into a differently-sized adapter.

Not attempted: showing rank as visually locked/disabled in the editor
the way Phase 5's checkboxes show/hide fields via `Port.visible_when`.
`visible_when` only ever compares against a *sibling Port's own widget
value* (`nodes/core.py`), evaluated client-side before the graph ever
runs -- but whether `resources.continue_lora_sd` is set isn't a Port's
own value at all, it's a property of whatever real Python object ends
up wired into `resources`, which doesn't exist until the graph actually
executes that far. Same reason this node has no `diagnostics()`
override either: Phase 5's live diagnostics (the `/diagnostics`
endpoint, `nodegraph.js`'s `scheduleDiagnostics()`) send plain,
JSON-safe widget values (`{params: {...}}`) -- `resources` being a
wired, non-JSON object rather than a path string is exactly what
that mechanism can't reach. The enforcement here is real (this file's
own `build()` below); "the editor already told you rank doesn't matter
here before you ran anything" is not -- a real, disclosed limitation
of what's built today, not an oversight.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..core import Node, Port
from .lora_training_resources import (
    LoRATrainingResources,
    LoRATrainingSkeleton,
    SDXL_LoraTrainer,
    SDXL_LoRATrainingResources,
    _lora_rank,
)

# One entry per (task, architecture) pair -- mirrors
# resources_controller.py's own _PRESETS in spirit (grow by adding an
# entry, not by rewriting this node), simpler in shape here because
# dispatch is automatic (the wired resources' own type) rather than a
# user-facing selector (there's nothing to choose: the architecture was
# already decided by whichever ResourcesControllerNode preset produced
# this specific resources object).
_TRAINER_FOR_RESOURCES: dict[type, type[LoRATrainingSkeleton]] = {
    SDXL_LoRATrainingResources: SDXL_LoraTrainer,
}

_UNET_WEIGHT_STORE_CHOICES = ("bf16", "nf4")


class LoRATrainingConfigNode(Node):
    """LoRA injection, configured -- see this module's own docstring
    for the full split with `ResourcesControllerNode`. `resources` is a
    wired `LoRATrainingResources` input (Phase 5's own output, not a
    path/widget this node resolves itself -- there's nothing left to
    load, `resources` already holds real, loaded objects).

    `unet_weight_store` mirrors `ResourcesControllerNode`'s own retired
    `unet_weight_store` Port exactly (same choices, same meaning) --
    moved here rather than reinvented, because storage of the *frozen*
    base weights is only meaningful once something is actually being
    injected around them, which happens here, not in Phase 5.
    """

    INPUTS: ClassVar[dict[str, Port]] = {
        "resources": Port(
            name="resources", type=LoRATrainingResources, required=True,
            doc="Phase 5's own output (ResourcesControllerNode) -- verified, "
                "not yet LoRA-injected.",
        ),
        "rank": Port(
            name="rank", type=int, required=False, default=64,
            doc="Ignored when resources.continue_lora_sd is set -- see this "
                "module's own top docstring. Free choice otherwise.",
        ),
        "alpha": Port(name="alpha", type=float, required=False, default=1.0),
        "unet_weight_store": Port(
            name="unet_weight_store", type=str, required=False, default="bf16",
            choices=_UNET_WEIGHT_STORE_CHOICES,
            doc="Frozen UNet base-weight storage. 'nf4' quantizes to ~4 bits/parameter "
                "(nodes/model/nf4_weight_store.py) -- real VRAM savings, real "
                "quantization error, genuinely lossy before any training happens.",
        ),
    }

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "trainer": Port(
            name="trainer", type=LoRATrainingSkeleton, required=True,
            doc="Real, LoRA-injected .unet/.clip/.vae_sd/.lora attributes, ready to "
                "train. What consumes this (TrainerNode and friends) is still open -- "
                "see docs/resources_controller_redesign_plan.md's own Phase 6 section.",
        ),
    }

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        resources = inputs["resources"]

        trainer_cls = _TRAINER_FOR_RESOURCES.get(type(resources))
        if trainer_cls is None:
            raise ValueError(
                f"{type(self).__name__}: no LoRATrainingSkeleton registered for "
                f"resources of type {type(resources).__name__} -- "
                f"_TRAINER_FOR_RESOURCES in this file needs a new entry for it."
            )

        rank = inputs.get("rank", self.INPUTS["rank"].default)
        if resources.continue_lora_sd is not None:
            detected_rank = _lora_rank(resources.continue_lora_sd)
            if detected_rank is None:
                # Shouldn't be reachable in practice -- ResourcesControllerNode's own
                # validators already reject a continue_lora_path whose modules don't
                # agree on one rank before this object could ever exist. Checked again
                # here anyway: this node doesn't get to assume what produced `resources`
                # actually enforced that, only that its own contract (a raw state dict)
                # was honored.
                raise ValueError(
                    "resources.continue_lora_sd's own modules don't agree on a single "
                    "rank -- can't auto-configure rank from it."
                )
            rank = detected_rank  # the whole point: not a value `inputs["rank"]` gets
            # a say in once there's a real continuing LoRA to match instead.

        unet_weight_store = inputs.get(
            "unet_weight_store", self.INPUTS["unet_weight_store"].default)
        if unet_weight_store == "nf4":
            from .nf4_weight_store import NF4WeightStore
            weight_store_factory = NF4WeightStore
        else:
            weight_store_factory = None  # inject_lora()'s own default: BF16WeightStore

        trainer = trainer_cls.from_resources(
            resources,
            rank=rank,
            alpha=inputs.get("alpha", self.INPUTS["alpha"].default),
            frozen_weight_store_factory=weight_store_factory,
        )
        result = {"trainer": trainer}
        self.validate_outputs(result)
        return result
