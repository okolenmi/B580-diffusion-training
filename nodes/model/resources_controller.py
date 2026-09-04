"""ResourcesController: the node from the original sketch, for real --
docs/resources_controller_redesign_plan.md's Phase 5. One place that
owns loading a base checkpoint, per-resource dtype detection/override,
and constructing a ready-to-train resources object -- replacing what
that plan's own "Why this exists" section calls "today's fully eager,
dtype-blind loader nodes."

Two pieces, deliberately kept separate rather than one class doing
everything:

ResourcePreset -- one concrete (task, architecture) combination (that
plan's own "Open design question" section: the Task axis x Architecture
axis compose into one preset per pair). Not a Node itself -- a plain
ResourcesControllerNode below holds one or more and delegates to
whichever one its own `preset` Port selects, so more than one preset
can share one Node class (NODE_KIND == "dynamic", Phase 3's own
suggestion-menu machinery, built exactly for this: "each preset of a
dynamic node counts as its own separate searchable entry"). Four pieces
per the settled ResourcePreset interface contract table in that
document's "Open design question" section -- three of which turned out,
once Port.choices actually existed (this same redesign's Consolidation
section, landed just before this), to need no new machinery at all:

  - list of inputs        -> self.inputs: dict[str, Port], the same
                              Port objects Node.INPUTS already uses.
  - parameter-value dict   -> Port.choices directly: node-facing
                              "available choices for the dropdown" IS a
                              choices-Port's own .choices; processor-
                              facing "the resolved single choice" IS
                              build()'s own already-validated
                              inputs[name]. No separate dict-shaped
                              mechanism needed -- this is exactly the
                              convergence the Consolidation section
                              identified between 11.4 and this table's
                              own row, now real rather than theoretical.
  - processor method       -> self.process(inputs), Node.build()'s real
                              logic, one level down so more than one
                              preset can share one Node class.
  - validators             -> genuinely new: self.validators, a
                              dict[str, Callable[[Any], list[str]]]
                              keyed by input name. Diagnostic text
                              ("UNET dtype: bf16"), not pass/fail --
                              Port.choices already rejects an outright
                              invalid dropdown value; this is for
                              things no closed choice list can check
                              (does this checkpoint file actually look
                              like SDXL, what dtype does it already
                              have, does this LoRA's rank match).

Only one concrete preset exists today -- LoRASDXLPreset, LoRA training
on SDXL -- matching "only one available for start" from the original
sketch description. A second (a different architecture, or a different
task like distillation) is a second ResourcePreset instance registered
alongside it in _PRESETS below, not a rewrite of any of this.

Revised twice against actual review, not just this document's own prose
description of the sketch (which drifted from it): a second pass moved
checkpoint loading out to a wired `ModelWeights` socket
(`SafetensorsCheckpointNode`'s own output), reasoning that the sketch's
own "o" input with a wire running into it meant this node shouldn't
resolve/load a path itself. Corrected back: this node resolves and
loads `checkpoint_path` itself (a `path_kind="checkpoint"` Port) --
that's the actual intent, a wire doesn't remove a string/path from the
picture either way, it just moves it to a different node for no real
gain here. The reuse-Phase-1's-`ModelWeights` benefit that motivated
the wire wasn't actually lost by reverting it: `_checkpoint_validator`/
`_inspect_checkpoint` below still construct a `ModelWeights` internally
from the resolved path rather than a second, hand-rolled header read --
same reuse, self-contained instead of externalized. Both passes did
correctly land `continue_training`/`frozen_lora` as real `bool`
checkbox Ports gating their own path (and, for frozen, strength) inputs
-- not inferred from whether a path happens to be `None`, which is what
the very first pass did and is exactly the kind of implicit,
easy-to-misread state a checkbox exists to avoid; that stayed.

Also added on this same pass, all three real, all generic rather than
Resources-Controller-specific (same "build the mechanism once" posture
`Port.choices` itself already established): `Port.visible_when`
(nodes/core.py) -- `continue_lora_path`/`frozen_lora_path`/
`frozen_lora_strength` below are only shown by the editor while their
own gating checkbox reads True, the sketch's own "extra fields spawn
after checkbox state changes" ask -- a UI hint only, `process()` below
still enforces the real invariant since `Node`/`Port` themselves never
read this field (see its own docstring for why). `Node.diagnostics()`
(nodes/core.py) -- promoted from a Resources-Controller-only method to
a real, generic, overridable-per-node method with a `{}` default, so a
future live-inspection endpoint can call it on any node uniformly
(`NodeInfo.has_diagnostics` in `server/nodegraph_introspect.py` reports
whether a given class actually overrode it, the same is-this-actually-
overridden check `NODE_KIND == "dynamic"` already needs for
`list_presets()`) -- the sketch's own "node works with the server,
calculates values, shows extra things" ask; the endpoint and
`nodegraph.js` wiring to actually call it live are still the deferred
piece below, but the mechanism itself is real now, not
Resources-Controller-specific plumbing bolted on ad hoc.
`LoRATrainingSkeleton.describe()`
(nodes/model/lora_training_resources.py) -- a real, universal,
read-only summary (dtype/footprint per component) built entirely out of
that class's own existing `DeviceResident`/`TrainableModel` methods, so
a future node consuming this preset's `resources` output (Phase 6) has
a real interface to call rather than reaching into `.unet`/`.clip`/
`.vae_sd` attributes directly -- "some methods to get data from items...
a universal interface other nodes may use later," checked against what
already existed first (`per_resident_footprint_bytes()`,
`trainable_parameters()`) rather than duplicating any of it.
`unet_dtype` also gained a real `"inherited"` choice (not merely a
label) -- resolves to the attached checkpoint's own detected UNet dtype
at `process()` time, matching the sketch's own bottom-table default for
two of its four rows, using the exact same header-read
`_inspect_checkpoint()` below already does for the "doesn't look like
SDXL" check, not a second detection pass.

Not built here, honestly deferred rather than silently missing: the
actual live query endpoint (mirroring Phase 2's own existing
`/nodegraph/assets/{kind}/inspect`) for the editor to call
`diagnostics()` as the user attaches/edits a resource, and the
`nodegraph.js` wiring both to call it and to render `visible_when`
(hide/show a Port's own row live, not just at initial spawn) -- the
sketch's actual "validity indicators" and "fields spawn after checkbox
state" UX. Both mechanisms above (`Node.diagnostics()`,
`Port.visible_when`) are real and already introspectable through
`node_info_to_dict()` today; what's missing is only the server route
and the two matching pieces of `nodegraph.js` rendering logic, matching
how every phase before this one sequenced backend-before-frontend.
Also not built: dynamically adding a choice to an already-spawned
node's dropdown from a live server response (the sketch's own
"inherited" example of this) -- `unet_dtype` above sidesteps the need
for that specifically by making `"inherited"` a real, always-present
static choice instead, which needed no new frontend machinery; a
genuinely dynamic per-instance choice-list extension is a separate,
larger frontend feature, not attempted here.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from ..core import Node, NodePreset, Port
from ..components.layout import ProjectLayout
from .handle import ModelWeights
from .lora_training_resources import LoRATrainingSkeleton, SDXL_LoraTrainer
from .resource_inspection import dtype_to_str, inspect_lora, str_to_dtype


class ResourcePreset(ABC):
    """One concrete (task, architecture) combination -- see this
    module's own docstring for the full contract. Not a Node; a
    ResourcesControllerNode instance holds one or more of these."""

    name: ClassVar[str]
    inputs: ClassVar[dict[str, Port]]
    outputs: ClassVar[dict[str, Port]]
    validators: ClassVar[dict[str, Callable[[Any], list[str]]]] = {}
    # Per-input diagnostic text -- see module docstring. Not every input
    # needs an entry; a plain numeric knob (rank, alpha, strength) has
    # nothing worth inspecting and just doesn't appear here.

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return  # still abstract -- same exemption Node.__init_subclass__ gives
        for attr in ("name", "inputs", "outputs"):
            if not hasattr(cls, attr):
                raise TypeError(
                    f"{cls.__name__}: must set class attribute {attr!r} -- see "
                    f"ResourcePreset's own docstring for the full contract."
                )
        if not isinstance(cls.inputs, dict) or not isinstance(cls.outputs, dict):
            raise TypeError(f"{cls.__name__}.inputs/outputs must be dicts.")
        if not cls.outputs:
            raise TypeError(
                f"{cls.__name__} declares no outputs -- same requirement "
                f"Node.__init_subclass__ enforces for OUTPUTS, for the same reason."
            )

    def node_preset(self) -> NodePreset:
        """This preset's own required-only inputs/outputs, for
        ResourcesControllerNode.list_presets() -- derived from
        self.inputs/self.outputs rather than hand-duplicated, so the
        two can't drift (NodePreset itself already rejects a
        required=False Port inside required_inputs/required_outputs,
        so this filters by construction rather than by promise)."""
        return NodePreset(
            name=self.name,
            required_inputs={k: p for k, p in self.inputs.items() if p.required},
            required_outputs={k: p for k, p in self.outputs.items() if p.required},
        )

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        """{input_name: [diagnostic lines]} for every input that has a
        validator and a value actually given -- the sketch's "UNET
        dtype: bf16" rows. Skips an input with no validator or no
        value (nothing to inspect). A validator raising here (a
        genuinely corrupt/wrong-shaped file, say) becomes a
        ["ERROR: ..."] line for that one input rather than aborting
        the whole batch -- appropriate for a "show me everything that's
        wrong across every attached resource" UI-facing method. Contrast
        process() below, which wants exactly the opposite (fail loudly,
        don't downgrade a real problem into display text) for the same
        underlying check."""
        result = {}
        for name, validate in self.validators.items():
            if name not in inputs or inputs[name] is None:
                continue
            try:
                result[name] = validate(inputs[name])
            except Exception as e:
                result[name] = [f"ERROR: {e}"]
        return result

    @abstractmethod
    def process(self, inputs: dict) -> dict[str, Any]:
        """Node.build()'s real logic, given already-validate_inputs()-
        checked, already-choice-checked values (ResourcesControllerNode
        below calls self.validate_inputs(inputs) before this ever
        runs). Returns a dict covering this preset's own self.outputs
        keys."""


def _inspect_checkpoint(relative_path: str):
    """Resolves + wraps in ModelWeights + raises the "doesn't look like
    SDXL" check -- the one real implementation _checkpoint_validator
    (text for diagnostics()) and process()'s "inherited" dtype
    resolution both need, so neither one parses the other's own
    display-formatted string for structured data (a real, if minor,
    footgun -- caught while writing this the second way round)."""
    layout = ProjectLayout.from_paths_module()
    resolved = layout.resolve_safe_model_path(relative_path, "checkpoint")
    dtypes = ModelWeights(resolved).inspect_dtypes()
    if dtypes["unet"].key_count == 0:
        raise ValueError(
            f"{resolved}: no keys classified as UNet (expected a "
            f"'model.diffusion_model.' prefix) -- doesn't look like an SDXL checkpoint."
        )
    return dtypes


def _checkpoint_validator(relative_path: str) -> list[str]:
    """UNET/CLIP/VAE dtype, one line each, matching the sketch's own
    "UNET dtype: bf16" row shape -- ModelWeights.inspect_dtypes() (Phase
    1, header-only, doesn't touch tensor data) via _inspect_checkpoint()
    above, not a second implementation of the same header read. This is
    reachable from the graph editor over the network the same as any
    path_kind Port, "don't trust the client" applies here too, not just
    inside build()."""
    dtypes = _inspect_checkpoint(relative_path)
    lines = []
    for component in ("unet", "clip", "vae"):
        info = dtypes[component]
        label = dtype_to_str(info.dtype) or ("mixed" if info.key_count else "absent")
        lines.append(f"{component.upper()} dtype: {label} ({info.key_count} tensors)")
    return lines


def _lora_validator(relative_path: str) -> list[str]:
    """Same sandboxing/raise-vs-diagnostic reasoning as
    _checkpoint_validator above, for a saved LoRA (kind="lora") --
    shared by both frozen_lora_path and continue_lora_path, since both
    are the same file format (lora_phases.py's lora_key())."""
    layout = ProjectLayout.from_paths_module()
    resolved = layout.resolve_safe_model_path(relative_path, "lora")
    info = inspect_lora(resolved)
    if info.key_count == 0:
        raise ValueError(f"{resolved}: no lora_down.weight keys found -- not a saved LoRA file.")
    dtype_label = dtype_to_str(info.dtype) or "mixed"
    rank_label = info.rank if info.rank is not None else "mixed"
    return [f"LoRA dtype: {dtype_label}, rank: {rank_label} ({info.key_count} modules)"]


# "inherited" resolves to whatever the attached checkpoint's own UNet dtype
# already is (process() below) -- a real fourth choice, not merely a label,
# and the default: matches the sketch's own bottom-table default of
# "<inherited>" for two of its four rows, and means picking a preset and
# attaching a checkpoint is enough to get a sensible run without also having
# to already know that checkpoint's own dtype.
_UNET_DTYPE_CHOICES = ("inherited", "bfloat16", "float16", "float32")
_UNET_WEIGHT_STORE_CHOICES = ("bf16", "nf4")


class LoRASDXLPreset(ResourcePreset):
    """LoRA training on SDXL -- the one concrete preset that exists
    today. Wraps SDXL_LoraTrainer (nodes/model/lora_training_resources.py,
    already real, already Phase-4-verified) rather than reimplementing
    any of its construction pipeline (split -> merge frozen LoRA ->
    inject -> mask CLIP -> load continue-from LoRA).

    Deliberately narrower than SDXL_LoraTrainer's own full constructor:
    scaling_policy/resource_policy/adapter_strategy/target_modules all
    stay at their real defaults (ClassicLoRAScaling, no resource
    policy, PlainLoRAAdapter, every Linear/Conv2d target) -- a preset's
    whole point is sensible defaults, not exposing every knob
    SDXL_LoraTrainer accepts. Anyone who needs those stays on the
    existing manual path (ComfyUNetLoRANode + friends), unaffected by
    this -- same "advanced path stays available" story this redesign's
    own Consolidation section already settled for that node.

    checkpoint_path is a path_kind="checkpoint" Port this node resolves
    and loads itself, matching the original hand-drawn sketch's own
    intent (a self-contained node, not one that pushes checkpoint
    loading out to a wire) -- an earlier pass here tried a wired
    ModelWeights socket instead and was corrected back. Reuses Phase 1's
    ModelWeights class internally either way (see _checkpoint_validator
    above) rather than a second, hand-rolled path-resolve-and-inspect,
    so nothing about that reuse was actually lost by not wiring it.
    project_layout stays internal (ProjectLayout.from_paths_module())
    for the same reason as before: every real path input's own
    validator needs the exact same resolution process() itself uses,
    and threading a per-call override through the settled
    single-argument validate(value) signature isn't a natural fit.

    continue_training/frozen_lora are real bool Ports (checkboxes),
    gating continue_lora_path/frozen_lora_path(+frozen_lora_strength) --
    both structurally, via Port.visible_when (nodes/core.py; the graph
    editor hides the gated Port's own row while its checkbox reads
    False, matching the sketch's own collapsed "Frozen LoRA" row), and
    semantically, via process()'s own both-directions check below
    (checked-without-a-path and a-path-without-being-checked both raise
    a clear error) -- visible_when is a UI hint only, never enforced by
    Node/Port themselves (see that field's own docstring), so process()
    still has to check this for real.

    Two dtype axes exposed, both wired all the way through to real code
    today (Phase 4's inject_lora()/build_lora_injected_unet()):
    unet_dtype (compute dtype, including the real "inherited" choice --
    see _UNET_DTYPE_CHOICES above) and unet_weight_store (frozen-weight
    storage, training_pipeline_design.md section 11.3 item 1). CLIP and
    VAE dtype are diagnostics-only (validators above), not override
    Ports: SDXLArchitecture.build_text_encoder()'s own docstring already
    flags that SDXLClipEncoder hardcodes its dtype (no parameter exists
    to override it), and nothing in nodes/ converts vae_sd's dtype
    anywhere -- an override Port with nothing downstream to honor it
    would be a dishonest no-op, not a real knob. state_dtype (section
    11.3 item 2, optimizer momentum) is deliberately absent too --
    Consolidation section: "concretely unresolved," flagged for a real
    decision later, not implemented in isolation before that's settled.

    A fourth axis the original sketch's own bottom summary table also
    named -- "LoRA (training)" dtype, i.e. the trainable adapter's own
    parameter dtype, distinct from all of the above -- is real today but
    checked directly against core/lora.py and found NOT to be a knob at
    all: LoRALinear/LoRAConv2d hardcode `param_dtype = torch.float32`
    for lora_A/lora_B regardless of the frozen base's own dtype, with a
    detailed, load-bearing numerical justification in that file's own
    comment (bf16's ~7-8 bit mantissa silently rounds away small
    Adafactor updates at realistic LoRA-adapter magnitudes -- verified
    there down to "bit-for-bit unchanged after 2000 steps" at a
    realistic lr -- "every mainstream LoRA implementation" keeps this in
    fp32 for exactly that reason). Exposing this as an override Port
    would be worse than a no-op: a real, easy-to-reach footgun. It's
    correctly absent from inputs/validators/choices below, not an
    oversight -- LoRATrainingSkeleton.describe()'s own "lora_adapter"
    entry reports it as detected, read-only information instead.
    """

    name = "lora_sdxl"

    inputs: ClassVar[dict[str, Port]] = {
        "checkpoint_path": Port(
            name="checkpoint_path", type=str, required=True, path_kind="checkpoint",
            doc="Base SDXL checkpoint to train from. Relative to the configured "
                "checkpoints directory -- absolute paths and '..' are rejected.",
        ),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "unet_dtype": Port(
            name="unet_dtype", type=str, required=False, default="inherited",
            choices=_UNET_DTYPE_CHOICES,
            doc="UNet compute dtype -- inject_lora()'s own dtype kwarg. 'inherited' "
                "(the default) resolves to the attached checkpoint's own detected "
                "UNet dtype at build time -- not a placeholder, a real choice.",
        ),
        "unet_weight_store": Port(
            name="unet_weight_store", type=str, required=False, default="bf16",
            choices=_UNET_WEIGHT_STORE_CHOICES,
            doc="Frozen UNet base-weight storage. 'nf4' quantizes to ~4 bits/parameter "
                "(nodes/model/nf4_weight_store.py) -- real VRAM savings, real "
                "quantization error (~9% relative RMSE on realistic weight-like data), "
                "genuinely lossy before any training happens.",
        ),
        "rank": Port(name="rank", type=int, required=False, default=64),
        "alpha": Port(name="alpha", type=float, required=False, default=1.0),
        "continue_training": Port(
            name="continue_training", type=bool, required=False, default=False,
            doc="Load an existing saved LoRA into the freshly-injected trainable "
                "adapter and continue training from it, instead of starting fresh. "
                "Gates continue_lora_path below -- checked, that path is required; "
                "unchecked, it's ignored (and must be empty).",
        ),
        "continue_lora_path": Port(
            name="continue_lora_path", type=str, required=False, default=None, path_kind="lora",
            visible_when=("continue_training", True),
            doc="Only used when continue_training is checked. Different from "
                "frozen_lora_path below -- see LoRATrainingSkeleton.__init__'s own "
                "docstring for the distinction (this one stays trainable).",
        ),
        "frozen_lora": Port(
            name="frozen_lora", type=bool, required=False, default=False,
            doc="Merge an existing saved LoRA directly into the base weights before "
                "injection -- permanent, untrainable afterward. Gates "
                "frozen_lora_path/frozen_lora_strength below the same way "
                "continue_training gates continue_lora_path.",
        ),
        "frozen_lora_path": Port(
            name="frozen_lora_path", type=str, required=False, default=None, path_kind="lora",
            visible_when=("frozen_lora", True),
            doc="Only used when frozen_lora is checked.",
        ),
        "frozen_lora_strength": Port(
            name="frozen_lora_strength", type=float, required=False, default=1.0,
            visible_when=("frozen_lora", True),
            doc="Only used when frozen_lora is checked.",
        ),
    }

    outputs: ClassVar[dict[str, Port]] = {
        "resources": Port(
            name="resources", type=LoRATrainingSkeleton, required=True,
            doc="Real .unet/.clip/.vae_sd/.lora attributes, ready to train, plus a "
                "real describe() summary (dtype/footprint per component) -- see "
                "SDXL_LoraTrainer and LoRATrainingSkeleton.describe(). Phase 6 of "
                "docs/resources_controller_redesign_plan.md settles what consumes "
                "this; not built yet.",
        ),
    }

    validators: ClassVar[dict[str, Callable[[Any], list[str]]]] = {
        "checkpoint_path": _checkpoint_validator,
        "frozen_lora_path": _lora_validator,
        "continue_lora_path": _lora_validator,
    }

    def process(self, inputs: dict) -> dict[str, Any]:
        # Cheap, header-only check before the real (potentially multi-GB) load below --
        # same ordering SafetensorsCheckpointNode.build() already established, and the
        # only place this dtype info is computed twice (also drives diagnostics() for
        # the editor); doesn't cost a second full load, only a second header read.
        # Raw dict, not _checkpoint_validator's own formatted text -- see
        # _inspect_checkpoint()'s own docstring for why.
        checkpoint_dtypes = _inspect_checkpoint(inputs["checkpoint_path"])

        continue_training = inputs.get(
            "continue_training", self.inputs["continue_training"].default)
        continue_lora_path = inputs.get("continue_lora_path")
        if continue_training and continue_lora_path is None:
            raise ValueError("continue_training is checked but continue_lora_path wasn't given.")
        if not continue_training and continue_lora_path is not None:
            raise ValueError(
                "continue_lora_path was given but continue_training isn't checked -- "
                "check it, or clear the path."
            )

        frozen_lora = inputs.get("frozen_lora", self.inputs["frozen_lora"].default)
        frozen_lora_path = inputs.get("frozen_lora_path")
        if frozen_lora and frozen_lora_path is None:
            raise ValueError("frozen_lora is checked but frozen_lora_path wasn't given.")
        if not frozen_lora and frozen_lora_path is not None:
            raise ValueError(
                "frozen_lora_path was given but frozen_lora isn't checked -- "
                "check it, or clear the path."
            )

        from safetensors.torch import load_file

        layout = ProjectLayout.from_paths_module()
        checkpoint_sd = load_file(str(layout.resolve_safe_model_path(
            inputs["checkpoint_path"], "checkpoint")))

        frozen_lora_sd = None
        frozen_lora_strength = inputs.get(
            "frozen_lora_strength", self.inputs["frozen_lora_strength"].default)
        if frozen_lora:
            _lora_validator(frozen_lora_path)
            frozen_lora_sd = load_file(str(layout.resolve_safe_model_path(frozen_lora_path, "lora")))

        continue_lora_sd = None
        if continue_training:
            _lora_validator(continue_lora_path)
            continue_lora_sd = load_file(str(layout.resolve_safe_model_path(continue_lora_path, "lora")))

        unet_weight_store = inputs.get(
            "unet_weight_store", self.inputs["unet_weight_store"].default)
        if unet_weight_store == "nf4":
            from .nf4_weight_store import NF4WeightStore
            weight_store_factory = NF4WeightStore
        else:
            weight_store_factory = None  # build_lora_injected_unet()'s own default: BF16WeightStore

        unet_dtype = inputs.get("unet_dtype", self.inputs["unet_dtype"].default)
        if unet_dtype == "inherited":
            unet_component = checkpoint_dtypes["unet"]  # from the header check above,
            # not a second detection pass -- see _inspect_checkpoint()'s own docstring.
            if unet_component.dtype is None:
                raise ValueError(
                    "unet_dtype='inherited' but the checkpoint's own UNet tensors "
                    "aren't a single consistent dtype -- pick an explicit dtype instead."
                )
            unet_dtype = dtype_to_str(unet_component.dtype)

        trainer = SDXL_LoraTrainer(
            checkpoint_sd,
            device=inputs.get("device", self.inputs["device"].default),
            dtype=str_to_dtype(unet_dtype),
            rank=inputs.get("rank", self.inputs["rank"].default),
            alpha=inputs.get("alpha", self.inputs["alpha"].default),
            frozen_lora_sd=frozen_lora_sd,
            frozen_lora_strength=frozen_lora_strength,
            continue_lora_sd=continue_lora_sd,
            frozen_weight_store_factory=weight_store_factory,
        )
        return {"resources": trainer}


_PRESETS: dict[str, ResourcePreset] = {"lora_sdxl": LoRASDXLPreset()}
_DEFAULT_PRESET = "lora_sdxl"


class ResourcesControllerNode(Node):
    """The node from the original sketch -- see this module's own
    docstring. NODE_KIND == "dynamic": exactly one registered preset
    today (_PRESETS above), but the whole point of the Node/
    ResourcePreset split is more than one sharing this same class later
    (a different architecture, or a different task) without a rewrite
    -- see docs/resources_controller_redesign_plan.md's "Open design
    question" section for the Task x Architecture matrix this is built
    for.

    "preset selector" from that plan's own Phase 5 goal line: real,
    even with only one valid value today -- an ordinary Port.choices
    dropdown, not bespoke UI. INPUTS/OUTPUTS below are today's one
    preset's own inputs/outputs directly (NODE_KIND's own docstring on
    Node: "a dynamic node's *common* ports, present no matter which
    preset is chosen" -- with exactly one preset, its shape simply IS
    the common shape). Honest limitation, not hidden: a second preset
    with a genuinely different input/output shape needs real
    reconciliation here (a shared-subset INPUTS/OUTPUTS, plus the still-
    open Phase 3 item -- live per-preset shape resolution in the
    editor, `graph_executor.py`'s `_is_compatible()` resolving from
    `spec.params` -- before a spawned node's sockets could actually
    reshape to match a selected preset) -- deferred until a second
    preset actually exists, matching this whole redesign's own
    established practice of not generalizing before a second real case
    shows up.
    """

    NODE_KIND: ClassVar[str] = "dynamic"

    INPUTS: ClassVar[dict[str, Port]] = {
        "preset": Port(
            name="preset", type=str, required=False, default=_DEFAULT_PRESET,
            choices=tuple(_PRESETS),
            doc="Which (task, architecture) combination to build. Only one exists "
                "today -- see this class's own docstring.",
        ),
        **_PRESETS[_DEFAULT_PRESET].inputs,
    }
    OUTPUTS: ClassVar[dict[str, Port]] = _PRESETS[_DEFAULT_PRESET].outputs

    @classmethod
    def list_presets(cls) -> list[NodePreset]:
        return [preset.node_preset() for preset in _PRESETS.values()]

    def _selected_preset(self, inputs: dict) -> ResourcePreset:
        # `or`, not `.get(..., default)`: an explicit preset=None (e.g. a
        # direct Python call, not reachable through the editor's own
        # dropdown -- see buildChoicesWidget's own placeholder logic for
        # why a Port with a real default never shows one) should fall
        # back to the default the same as a genuinely missing key, not
        # raise a confusing KeyError from _PRESETS[None].
        name = inputs.get("preset") or self.INPUTS["preset"].default
        return _PRESETS[name]

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        """Per-input diagnostic text for whichever preset
        inputs["preset"] selects -- see ResourcePreset.diagnostics().
        Not called by build() itself. A live query path (mirroring
        Phase 2's own /nodegraph/assets/{kind}/inspect) for the editor
        to call this as the user attaches a resource, and the frontend
        display for it, are real, separate follow-up work -- see this
        module's own docstring."""
        return self._selected_preset(inputs).diagnostics(inputs)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        result = self._selected_preset(inputs).process(inputs)
        self.validate_outputs(result)
        return result
