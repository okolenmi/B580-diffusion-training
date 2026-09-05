"""ResourcesController: docs/resources_controller_redesign_plan.md's
Phase 5. A node interface over the basic functions needed to turn a
checkpoint (plus optional frozen/continue-training LoRAs) into a
ready-to-use, verified pack of resources for LoRA training -- and
nothing past that: not LoRA injection, not rank/alpha, not training
setup. Those are a separate, later node's job (Phase 6, not built yet)
-- see VerifiedResourcePack's own docstring in
nodes/model/lora_training_resources.py for the full reasoning, and this
file's own revision history at the bottom for how an earlier version of
this node got that boundary wrong.

Two pieces, deliberately kept separate rather than one class doing
everything:

ResourcePreset -- one concrete (task, architecture) combination (the
plan's own "Open design question" section: the Task axis x Architecture
axis compose into one preset per pair). Not a Node itself --
ResourcesControllerNode below holds one or more and delegates to
whichever one its own `preset` Port selects, so more than one preset
can share one Node class (NODE_KIND == "dynamic", Phase 3's own
suggestion-menu machinery, built exactly for this). Inputs and outputs
are meant to be the same shape for any future preset -- exactly four
resources, matching whatever architecture is underneath (unet, clip,
vae, continue_lora) -- so adding a second preset (a different
architecture, or a different task) is a second ResourcePreset instance
registered in _PRESETS below, not a rewrite of any of this.

Four pieces per the settled ResourcePreset interface contract table in
that document's "Open design question" section -- three of which need
no new machinery at all now that Port.choices exists (this same
redesign's Consolidation section):

  - list of inputs        -> self.inputs: dict[str, Port], the same
                              Port objects Node.INPUTS already uses.
  - parameter-value dict   -> Port.choices directly: node-facing
                              "available choices for the dropdown" IS a
                              choices-Port's own .choices; processor-
                              facing "the resolved single choice" IS
                              build()'s own already-validated
                              inputs[name].
  - processor method       -> self.process(inputs), Node.build()'s real
                              logic, one level down so more than one
                              preset can share one Node class.
  - validators             -> dict[str, Callable[[Any], list[str]]]
                              keyed by input name. Diagnostic text
                              ("UNET dtype: bf16"), not pass/fail --
                              Port.choices already rejects an outright
                              invalid dropdown value; this is for
                              things no closed choice list can check
                              (does this checkpoint file actually look
                              like SDXL, what dtype does it already
                              have, does this LoRA's rank look sane).
                              The "on the fly verification" this node
                              exists to do.

Also real, generic (not Resources-Controller-specific), living in
nodes/core.py: Port.visible_when (a Port's own row hidden by the editor
unless a named sibling Port holds a given value -- continue_lora_path/
frozen_lora_path/frozen_lora_strength below all use this against their
own gating checkbox) and Node.diagnostics() (a {}-default method any
node can override -- ResourcesControllerNode's own override below
delegates to whichever preset is selected; server/routes_nodegraph.py's
POST /nodegraph/node/{class_name}/diagnostics and matching
server/static/nodegraph.js wiring call it live as the editor's own
fields change).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from ..core import Node, NodePreset, Port
from ..components.layout import ProjectLayout
from .handle import ModelWeights
from .lora_training_resources import SDXLVerifiedResourcePack, VerifiedResourcePack
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
    # needs an entry; a plain numeric/bool knob has nothing worth
    # inspecting and just doesn't appear here.

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
        validator and a value actually given. Skips an input with no
        validator or no value (nothing to inspect). A validator raising
        here (a genuinely corrupt/wrong-shaped file) becomes a
        ["ERROR: ..."] line for that one input rather than aborting the
        whole batch -- appropriate for a "show me everything that's
        wrong across every attached resource" UI-facing method.
        Contrast process() below, which wants exactly the opposite
        (fail loudly, don't downgrade a real problem into display text)
        for the same underlying check."""
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
        checked, already-choice-checked values. Returns a dict covering
        this preset's own self.outputs keys."""


def _inspect_checkpoint(relative_path: str):
    """Resolves + wraps in ModelWeights + raises the "doesn't look like
    SDXL" check -- the one real implementation _checkpoint_validator
    (text for diagnostics()) and process()'s "inherited" dtype
    resolution both need, so neither parses the other's own
    display-formatted string for structured data."""
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
    """UNET/CLIP/VAE dtype, one line each -- ModelWeights.inspect_dtypes()
    (Phase 1, header-only, doesn't touch tensor data) via
    _inspect_checkpoint() above, not a second implementation of the
    same header read. Reachable from the graph editor over the network
    the same as any path_kind Port -- "don't trust the client" applies
    here too, not just inside build()."""
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
    are the same file format. Reports rank here purely as detected
    information -- nothing downstream in this node locks anything to
    it or otherwise acts on it; that's the future training node's job
    (see this module's own docstring)."""
    layout = ProjectLayout.from_paths_module()
    resolved = layout.resolve_safe_model_path(relative_path, "lora")
    info = inspect_lora(resolved)
    if info.key_count == 0:
        raise ValueError(f"{resolved}: no lora_down.weight keys found -- not a saved LoRA file.")
    dtype_label = dtype_to_str(info.dtype) or "mixed"
    rank_label = info.rank if info.rank is not None else "mixed"
    return [f"LoRA dtype: {dtype_label}, rank: {rank_label} ({info.key_count} modules)"]


# "inherited" resolves to whatever the attached checkpoint's own UNet dtype
# already is (process() below) -- a real choice, not merely a label, and the
# default: attaching a checkpoint is enough to get a sensible result without
# also having to already know that checkpoint's own dtype.
_UNET_DTYPE_CHOICES = ("inherited", "bfloat16", "float16", "float32")


class LoRASDXLPreset(ResourcePreset):
    """LoRA training on SDXL -- the one concrete preset that exists
    today. Wraps SDXLVerifiedResourcePack
    (nodes/model/lora_training_resources.py) rather than reimplementing
    any of its construction pipeline (merge frozen LoRA -> split ->
    dtype-convert -> build text encoder).

    Deliberately does NOT do LoRA injection, and so deliberately has no
    rank/alpha/frozen-weight-storage inputs at all -- see this module's
    own top docstring and VerifiedResourcePack's own docstring for why
    that's a real scope boundary, not an oversight. checkpoint_path is a
    path_kind="checkpoint" Port this node resolves and loads itself --
    a self-contained node, not one that pushes checkpoint loading out to
    a wire. project_layout stays internal
    (ProjectLayout.from_paths_module()): every real path input's own
    validator needs the exact same resolution process() itself uses,
    and threading a per-call override through the settled
    single-argument validate(value) signature isn't a natural fit.

    continue_training/frozen_lora are real bool Ports (checkboxes),
    gating continue_lora_path/frozen_lora_path(+frozen_lora_strength) --
    both structurally, via Port.visible_when (the editor hides the
    gated Port's own row while its checkbox reads False), and
    semantically, via process()'s own both-directions check below
    (checked-without-a-path and a-path-without-being-checked both raise
    a clear error) -- visible_when is a UI hint only, never enforced by
    Node/Port themselves, so process() still has to check this for
    real.

    unet_dtype is the one dtype axis exposed here, including the real
    "inherited" choice (see _UNET_DTYPE_CHOICES above) -- a plain
    tensor-conversion of the verified base weights, not an injection
    parameter (there's no injection at this stage to parameterize).
    CLIP and VAE dtype are diagnostics-only (validators above), not
    override Ports: SDXLArchitecture.build_text_encoder()'s own
    docstring already flags that SDXLClipEncoder hardcodes its dtype
    (no parameter exists to override it), and nothing in nodes/
    converts vae_sd's dtype anywhere -- an override Port with nothing
    downstream to honor it would be a dishonest no-op, not a real knob.
    """

    name = "lora_sdxl"

    inputs: ClassVar[dict[str, Port]] = {
        "checkpoint_path": Port(
            name="checkpoint_path", type=str, required=True, path_kind="checkpoint",
            doc="Base SDXL checkpoint to load. Relative to the configured "
                "checkpoints directory -- absolute paths and '..' are rejected.",
        ),
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "unet_dtype": Port(
            name="unet_dtype", type=str, required=False, default="inherited",
            choices=_UNET_DTYPE_CHOICES,
            doc="UNet dtype for the verified resource pack. 'inherited' (the default) "
                "resolves to the attached checkpoint's own detected UNet dtype at "
                "build time -- not a placeholder, a real choice.",
        ),
        "continue_training": Port(
            name="continue_training", type=bool, required=False, default=False,
            widget_only=True,
            doc="Carry an existing saved LoRA through as this pack's own continue_lora "
                "resource, for whatever node injects LoRA later to load into its "
                "freshly-created adapter instead of starting fresh. Gates "
                "continue_lora_path below.",
        ),
        "continue_lora_path": Port(
            name="continue_lora_path", type=str, required=False, default=None, path_kind="lora",
            visible_when=("continue_training", True),
            doc="Only used when continue_training is checked. Loaded and verified here "
                "(a real saved LoRA, real rank/dtype detected -- see diagnostics()), "
                "not injected here -- there's no adapter yet at this stage.",
        ),
        "frozen_lora": Port(
            name="frozen_lora", type=bool, required=False, default=False,
            widget_only=True,
            doc="Merge an existing saved LoRA directly into the base UNet weights -- "
                "permanent, no separate identity afterward, which is why it isn't one "
                "of this pack's own four output resources. Gates "
                "frozen_lora_path/frozen_lora_strength below.",
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
            name="resources", type=VerifiedResourcePack, required=True,
            doc="Exactly four things: .unet_sd, .clip, .vae_sd, .continue_lora_sd "
                "(None if continue_training wasn't checked) -- see "
                "VerifiedResourcePack's own docstring. NOT LoRA-injected -- that's a "
                "separate, later node's job (Phase 6 of "
                "docs/resources_controller_redesign_plan.md, not built yet).",
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

        unet_dtype = inputs.get("unet_dtype", self.inputs["unet_dtype"].default)
        if unet_dtype == "inherited":
            unet_component = checkpoint_dtypes["unet"]  # from the header check above,
            # not a second detection pass -- see _inspect_checkpoint()'s own docstring.
            if unet_component.dtype is None:
                raise ValueError(
                    "unet_dtype='inherited' but the checkpoint's own UNet tensors "
                    "aren't a single consistent dtype -- pick an explicit dtype instead."
                )
            resolved_dtype = unet_component.dtype
        else:
            resolved_dtype = str_to_dtype(unet_dtype)

        pack = SDXLVerifiedResourcePack(
            checkpoint_sd,
            device=inputs.get("device", self.inputs["device"].default),
            dtype=resolved_dtype,
            frozen_lora_sd=frozen_lora_sd,
            frozen_lora_strength=frozen_lora_strength,
            continue_lora_sd=continue_lora_sd,
        )
        return {"resources": pack}


_PRESETS: dict[str, ResourcePreset] = {"lora_sdxl": LoRASDXLPreset()}
_DEFAULT_PRESET = "lora_sdxl"


class ResourcesControllerNode(Node):
    """The Resources Controller -- see this module's own docstring.
    NODE_KIND == "dynamic": exactly one registered preset today
    (_PRESETS above), but the whole point of the Node/ResourcePreset
    split is more than one sharing this same class later (a different
    architecture, or a different task) without a rewrite -- see
    docs/resources_controller_redesign_plan.md's "Open design question"
    section for the Task x Architecture matrix this is built for.

    A real `preset` Port (choices=tuple(_PRESETS)) -- a dropdown, even
    with only one valid value today, via Port.choices, not bespoke UI.
    INPUTS/OUTPUTS below are today's one preset's own shape directly
    (a dynamic node's *common* ports, present no matter which preset is
    chosen -- with exactly one preset, its shape simply *is* the common
    shape); a second preset with a genuinely different shape needs real
    reconciliation here, deferred until a second preset actually
    exists. list_presets() delegates to each registered preset's own
    node_preset(). diagnostics(inputs) exposes each attached resource's
    validator output -- real and callable today, wired to a live
    server endpoint and nodegraph.js rendering (see this module's own
    top docstring).
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
        # `or`, not `.get(..., default)`: an explicit preset=None (e.g. a direct
        # Python call, not reachable through the editor's own dropdown) should fall
        # back to the default the same as a genuinely missing key, not raise a
        # confusing KeyError from _PRESETS[None].
        name = inputs.get("preset") or self.INPUTS["preset"].default
        return _PRESETS[name]

    def diagnostics(self, inputs: dict) -> dict[str, list[str]]:
        return self._selected_preset(inputs).diagnostics(inputs)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        result = self._selected_preset(inputs).process(inputs)
        self.validate_outputs(result)
        return result


# --- Revision history -------------------------------------------------------
# Landed in three real passes, not one -- kept short here rather than in a
# blow-by-blow across every docstring above, per direct feedback that the
# accumulated corrections themselves had become the confusing part:
#
# 1. First version: checkpoint_path as a path Port, rank/alpha/unet_dtype/
#    unet_weight_store as inputs, process() called SDXL_LoraTrainer(...)
#    directly (LoRA injection happened inside this node).
# 2. Corrected checkpoint_path to a wired ModelWeights socket, reasoning from
#    the sketch's own drawn wire -- then corrected back to a path Port
#    (direct feedback: a wire doesn't remove the string/path from the
#    picture, it just moves it for no real gain here). Added
#    continue_training/frozen_lora as real bool checkboxes with
#    Port.visible_when gating their paths (was inferring "enabled" from
#    whether a path was None). Added live diagnostics (Node.diagnostics(),
#    the /diagnostics endpoint, nodegraph.js wiring) and "inherited" as a
#    real unet_dtype choice.
# 3. Direct feedback that this node's whole scope was wrong: rank/alpha/
#    frozen-weight-storage are LoRA-injection specifics, not properties of a
#    verified resource, and belong on a separate, later training node (Phase
#    6) that doesn't exist yet -- e.g. that node, not this one, should size
#    a continuing LoRA's adapter to that LoRA's own real rank, a decision
#    this node has no business making. Removed rank/alpha/unet_weight_store
#    and the SDXL_LoraTrainer(...)/injection call entirely; this node now
#    produces VerifiedResourcePack (nodes/model/lora_training_resources.py,
#    new), not LoRATrainingSkeleton -- four verified-but-uninjected
#    resources, matching "for LoRA training specification there are only 4
#    objects: base unet, clip, vae, continue LoRA (optional)."
