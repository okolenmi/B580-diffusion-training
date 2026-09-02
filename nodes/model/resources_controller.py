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

Not built here, honestly deferred rather than silently missing: a live
query path (mirroring Phase 2's own existing
`/nodegraph/assets/{kind}/inspect`) for the editor to call
ResourcesControllerNode.diagnostics() as the user attaches a resource,
and the frontend display for it -- the sketch's actual "validity
indicators" UX. diagnostics() itself is real and callable today
(directly, or by a future such endpoint); what's missing is only the
route and the nodegraph.js wiring, matching how every phase before this
one sequenced backend-before-frontend.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from ..core import Node, NodePreset, Port
from ..components.layout import ProjectLayout
from .lora_training_resources import LoRATrainingSkeleton, SDXL_LoraTrainer
from .resource_inspection import dtype_to_str, inspect_checkpoint_dtypes, inspect_lora, str_to_dtype


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


def _checkpoint_validator(relative_path: str) -> list[str]:
    """UNET/CLIP/VAE dtype, one line each, matching the sketch's own
    "UNET dtype: bf16" row shape. Resolves the same sandboxed way
    process() below does (ProjectLayout.resolve_safe_model_path,
    "checkpoint") -- this is reachable from the graph editor over the
    network the same as any path_kind Port, "don't trust the client"
    applies here too, not just inside build(). Raises (not just a
    diagnostic line -- see ResourcePreset.diagnostics() for how a
    caller there turns this into one) when this doesn't look like an
    SDXL checkpoint at all: unet component entirely absent."""
    layout = ProjectLayout.from_paths_module()
    resolved = layout.resolve_safe_model_path(relative_path, "checkpoint")
    dtypes = inspect_checkpoint_dtypes(resolved)
    if dtypes["unet"].key_count == 0:
        raise ValueError(
            f"{resolved}: no keys classified as UNet (expected a "
            f"'model.diffusion_model.' prefix) -- doesn't look like an SDXL checkpoint."
        )
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


_UNET_DTYPE_CHOICES = ("bfloat16", "float16", "float32")
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

    project_layout stays internal (ProjectLayout.from_paths_module()),
    not an exposed override Port here: every real path input's own
    validator needs the exact same resolution process() itself uses,
    and threading a per-call override through the settled
    single-argument validate(value) signature isn't a natural fit --
    worth revisiting if a real need for it shows up specifically here,
    not speculatively now (SafetensorsCheckpointNode's own
    project_layout Port exists mainly for deterministic testing, not
    because end users routinely override it).

    Two dtype axes exposed, both wired all the way through to real code
    today (Phase 4's inject_lora()/build_lora_injected_unet()):
    unet_dtype (compute dtype) and unet_weight_store (frozen-weight
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
            name="unet_dtype", type=str, required=False, default="bfloat16",
            choices=_UNET_DTYPE_CHOICES,
            doc="UNet compute dtype -- inject_lora()'s own dtype kwarg.",
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
        "frozen_lora_path": Port(
            name="frozen_lora_path", type=str, required=False, default=None, path_kind="lora",
            doc="Optional saved LoRA merged directly into the base weights before "
                "injection -- permanent, untrainable afterward. None = no frozen LoRA "
                "(the common case).",
        ),
        "frozen_lora_strength": Port(
            name="frozen_lora_strength", type=float, required=False, default=1.0,
            doc="Only meaningful with frozen_lora_path set -- process() below flags the "
                "likely-mistake case of one given without the other.",
        ),
        "continue_lora_path": Port(
            name="continue_lora_path", type=str, required=False, default=None, path_kind="lora",
            doc="Optional saved LoRA loaded into the freshly-injected trainable adapter, "
                "continuing training from these weights rather than starting fresh. "
                "None = fresh start (the common case). Different from frozen_lora_path -- "
                "see LoRATrainingSkeleton.__init__'s own docstring for the distinction.",
        ),
    }

    outputs: ClassVar[dict[str, Port]] = {
        "resources": Port(
            name="resources", type=LoRATrainingSkeleton, required=True,
            doc="Real .unet/.clip/.vae_sd/.lora attributes, ready to train -- see "
                "SDXL_LoraTrainer. Phase 6 of docs/resources_controller_redesign_plan.md "
                "settles what consumes this; not built yet.",
        ),
    }

    validators: ClassVar[dict[str, Callable[[Any], list[str]]]] = {
        "checkpoint_path": _checkpoint_validator,
        "frozen_lora_path": _lora_validator,
        "continue_lora_path": _lora_validator,
    }

    def process(self, inputs: dict) -> dict[str, Any]:
        frozen_lora_path = inputs.get("frozen_lora_path")
        frozen_lora_strength = inputs.get(
            "frozen_lora_strength", self.inputs["frozen_lora_strength"].default)
        if frozen_lora_path is None and frozen_lora_strength != self.inputs["frozen_lora_strength"].default:
            raise ValueError(
                "frozen_lora_strength was given but frozen_lora_path wasn't -- strength "
                "only means something for a frozen LoRA actually being merged, likely a "
                "forgotten path rather than an intentional no-op value."
            )

        # Cheap, header-only check before the real (potentially multi-GB) load below --
        # same ordering SafetensorsCheckpointNode.build() already established, and the
        # only place this dtype info is used twice (also drives diagnostics() for the
        # editor); doesn't cost a second full load, only a second header read.
        _checkpoint_validator(inputs["checkpoint_path"])

        from safetensors.torch import load_file

        layout = ProjectLayout.from_paths_module()
        checkpoint_sd = load_file(str(layout.resolve_safe_model_path(
            inputs["checkpoint_path"], "checkpoint")))
        frozen_lora_sd = None
        if frozen_lora_path is not None:
            _lora_validator(frozen_lora_path)
            frozen_lora_sd = load_file(str(layout.resolve_safe_model_path(frozen_lora_path, "lora")))
        continue_lora_path = inputs.get("continue_lora_path")
        continue_lora_sd = None
        if continue_lora_path is not None:
            _lora_validator(continue_lora_path)
            continue_lora_sd = load_file(str(layout.resolve_safe_model_path(continue_lora_path, "lora")))

        unet_weight_store = inputs.get(
            "unet_weight_store", self.inputs["unet_weight_store"].default)
        if unet_weight_store == "nf4":
            from .nf4_weight_store import NF4WeightStore
            weight_store_factory = NF4WeightStore
        else:
            weight_store_factory = None  # build_lora_injected_unet()'s own default: BF16WeightStore

        trainer = SDXL_LoraTrainer(
            checkpoint_sd,
            device=inputs.get("device", self.inputs["device"].default),
            dtype=str_to_dtype(inputs.get("unet_dtype", self.inputs["unet_dtype"].default)),
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
