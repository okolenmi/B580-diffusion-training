"""Pure introspection for the node-graph playground (dev/testing tab).

Deliberately mirrors config_schema.py's core principle: UI metadata is
*derived* from the real Python class at import time, never hand-duplicated
in a separate file. That's a structural choice, not a style preference --
hand-authored, separately-maintained UI metadata can drift out of sync
with what the underlying schema actually supports. A node graph whose
port list comes from inspect.signature() on the real class cannot drift
the same way: change the class, the graph's rendering changes with it,
automatically, with no second file to remember to update.

This module has ZERO side effects and ZERO coupling to the rest of the
codebase beyond importing classes to introspect. It doesn't execute
anything, doesn't touch config, doesn't affect the production training
path in any way. Safe to import, safe to iterate on, safe to delete.
"""

from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PortInfo:
    name: str
    type_str: str          # best-effort human-readable type (from annotation, or "any" if untyped)
    default: str | None     # repr() of the default, or None if required (no default)
    required: bool
    type_mro: list[str]     # [type_str, ...its base classes...], "Any" alone for typing.Any -- lets
                             # a caller check "is this output's type a subclass of that input's type"
                             # instead of comparing type_str for exact equality, e.g. FusedOptimizerHandle
                             # (a real OptimizerHandle subclass) connecting into an OptimizerHandle input.
    doc: str = ""            # Port.doc, the tooltip text -- empty for legacy-guessed ports (no such field to read)
    path_kind: str | None = None  # Port.path_kind -- tells the UI to render a file/folder picker
    choices: list[str] | None = None  # Port.choices -- tells the UI to render a dropdown
    # instead of freeform text; None for a legacy-guessed port (no Port object to read) or
    # any real Port that didn't declare one. list, not Port's own tuple: this dataclass's
    # fields are JSON-serialized (node_info_to_dict() below), and type_mro already sets the
    # precedent of "list" for this same kind of value here.
    visible_when: list | None = None  # Port.visible_when as [other_port_name, value] --
    # tells the UI to hide this Port's own row unless the sibling input named here
    # currently holds exactly `value`. list (JSON has no tuple), same reasoning as choices
    # above; None for a legacy-guessed port or any real Port that didn't declare one.
    widget_only: bool = False  # Port.widget_only -- tells the UI to draw only this
    # Port's own widget, no wire socket at all. False for a legacy-guessed port (no
    # Port object to read, and the safer default: showing a wire socket that turns out
    # unneeded is a minor visual redundancy, hiding one that was actually needed is a
    # real capability loss) or any real Port that didn't set it.


@dataclass
class PresetInfo:
    """Introspected form of nodes.core.NodePreset -- see that class's
    own docstring. Only ever appears on a NodeInfo whose node_kind is
    "dynamic"."""
    name: str
    required_inputs: list[PortInfo]
    required_outputs: list[PortInfo]


@dataclass
class NodeInfo:
    class_name: str
    module: str
    doc: str                # first line of the class docstring, or ""
    inputs: list[PortInfo]  # derived from __init__'s parameters (minus self)
    outputs: list[PortInfo] # see introspect_class()'s docstring for how these are derived
    bases: list[str] | None = None  # real inheritance chain (Node subclasses only) -- None for legacy-class introspection, where there's no such formal chain to report
    display_name: str = ""  # palette label -- Node.DISPLAY_NAME if the class set one,
    # else auto-derived by _auto_display_name() below. Always non-empty for any real
    # NodeInfo (see introspect_node_class()/introspect_legacy_class()); the "" default
    # here exists only so callers constructing a NodeInfo by hand in a test aren't
    # forced to pass it. class_name (above) is untouched by any of this -- it stays the
    # literal __name__ that saved graphs resolve against. See design doc section 11.5.
    node_kind: str = "static"  # nodes.core.Node.NODE_KIND, verbatim -- "static" for every
    # node in this project today, legacy-guessed classes included (introspect_legacy_class()
    # has no NODE_KIND concept to read, so it's always "static" there, correctly: nothing
    # legacy-introspected has presets).
    presets: list[PresetInfo] | None = None  # only non-None when node_kind == "dynamic" --
    # see docs/resources_controller_redesign_plan.md Phase 3 for what this is for
    # (the editor's suggestion-menu search) and nodes.core.NodePreset for why it's
    # required-ports-only, not a full shape resolution.
    has_diagnostics: bool = False  # nodes.core.Node.diagnostics() actually overridden,
    # not just inherited -- lets a caller (a future live-inspection endpoint/the editor
    # calling it) know whether this class has anything to say at all before ever
    # bothering to call it, the same is-this-actually-overridden check node_kind ==
    # "dynamic" already needs for list_presets(). Always False for a legacy-guessed
    # class -- there's no real class body there to have overridden anything.


# Domain vocabulary this project's own class names actually use that a generic
# capital-letter split gets wrong -- "ComfyUNetLoRA" split purely on capital
# boundaries gives "Comfy U Net Lo R A", not "Comfy UNet LoRA". Checked
# case-sensitively, longest-first, at each position before falling back to a
# plain [A-Z][a-z0-9]* word. This is deliberately a closed, curated list
# grounded in real class names in nodes/ today (see server/nodegraph_registry.py's
# _load() for the full set this was derived against) -- not an attempt at general
# PascalCase/acronym parsing. A future class whose name introduces a new token
# this list doesn't know about will still get *a* label (each of its capitals
# split into its own word, same as any generic splitter's fallback) -- reads
# awkwardly rather than crashing, and is exactly what Node.DISPLAY_NAME exists
# to override.
_KNOWN_DISPLAY_TOKENS = sorted(
    ["UNet", "LoRA", "DoRA", "QDoRA", "CAME", "AdamW", "SDXL", "SNR", "NF4",
     "BF16", "XPU", "VRAM", "LR", "P2"],
    key=len, reverse=True,
)

_WORD_RE = re.compile(r"[A-Z][a-z0-9]*")


def _auto_display_name(class_name: str) -> str:
    """Auto-derive a palette label: "ComfyUNetLoRANode" -> "Comfy UNet LoRA".
    Strips a trailing "Node" suffix (present on every real Node subclass,
    carries no information --
    everything in the palette is a node), then splits what's left into
    words: a known domain token (_KNOWN_DISPLAY_TOKENS) if one matches at
    the current position, else one capital letter followed by any run of
    lowercase letters/digits. Never raises -- worst case for an unrecognized
    run of capitals is one word per capital, not a crash."""
    stem = class_name[:-4] if class_name.endswith("Node") and len(class_name) > 4 else class_name
    words: list[str] = []
    i = 0
    while i < len(stem):
        for token in _KNOWN_DISPLAY_TOKENS:
            if stem.startswith(token, i):
                words.append(token)
                i += len(token)
                break
        else:
            m = _WORD_RE.match(stem, i)
            if m:
                words.append(m.group())
                i = m.end()
            else:
                # Not a capital-start position (shouldn't happen for real
                # PascalCase class names) -- consume one char rather than
                # loop forever, still produces *a* label.
                words.append(stem[i])
                i += 1
    return " ".join(words) if words else class_name


def _display_name_for(cls: type) -> str:
    override = getattr(cls, "DISPLAY_NAME", None)
    return override if override else _auto_display_name(cls.__name__)


def _type_str(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _type_mro(t: Any) -> list[str]:
    if t is Any or t is inspect.Parameter.empty:
        return ["Any"]
    if isinstance(t, type):
        return [c.__name__ for c in t.__mro__ if c.__name__ not in ("object", "ABC")]
    return [str(t)]


def introspect_legacy_class(cls: type, category: str | None = None) -> NodeInfo:
    """Derive a NodeInfo by GUESSING from cls.__init__'s real signature --
    for classes that were never designed with a node interface in mind
    (i.e. anything still living under core/, manager/, server/ that hasn't
    been wrapped in nodes/ yet). See introspect_node_class() below for the
    strictly-better alternative used for anything that HAS been migrated:
    reading real, declared Port metadata instead of guessing from a
    constructor signature. Kept under this more explicit name (renamed from
    introspect_class()) specifically so it's obvious at the call site which
    kind of introspection -- guessed vs. declared -- a given endpoint is
    doing; conflating the two under one name was a real risk of someone
    reasonably assuming a class's presence here meant it had a real,
    declared contract when it might only have a guessed one.

    No modification to cls, no instantiation -- read-only introspection.

    Deliberate boundary between what's structurally derived vs. supplied:
    INPUTS are 100% derived from the real signature -- that's a structural
    fact about the class, cannot drift, no second file involved. OUTPUTS
    are different in kind: the standardized rule this codebase is adopting
    is "a node that wraps a constructor has exactly one output: an instance
    of that class" -- but the class itself can't self-report its own
    semantic ROLE in a pipeline. That role has to come from the calling
    context that already knows the domain. So: pass `category` explicitly
    when the caller knows it, and get a real, typed output port back.
    Don't pass it, and outputs comes back empty -- explicitly, rather than
    fabricating a guessed label.
    """
    doc = (inspect.getdoc(cls) or "").strip().split("\n")[0]
    sig = inspect.signature(cls.__init__)
    ports = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue  # *args / **kwargs aren't real named ports
        has_default = param.default is not inspect.Parameter.empty
        ports.append(PortInfo(
            name=name,
            type_str=_type_str(param.annotation),
            default=repr(param.default) if has_default else None,
            required=not has_default,
            type_mro=_type_mro(param.annotation),
        ))
    outputs = []
    if category is not None:
        outputs.append(PortInfo(
            name=category,
            type_str=cls.__name__,
            default=None,
            required=True,
            type_mro=_type_mro(cls),
        ))
    return NodeInfo(
        class_name=cls.__name__,
        module=cls.__module__,
        doc=doc,
        inputs=ports,
        outputs=outputs,
        bases=None,  # no formal Node inheritance chain to report for a guessed class
        display_name=_display_name_for(cls),
    )


def _port_type_str(t: Any) -> str:
    """Like _type_str, but for a nodes.core.Port's .type field, which is a
    real Python type/class (or typing.Any) rather than an inspect.Parameter
    annotation -- close enough in shape to share most of the logic, but
    kept separate since the two are conceptually different inputs (a
    declared contract vs. a guessed one) and might need to diverge later."""
    if t is Any:
        return "any"
    if hasattr(t, "__name__"):
        return t.__name__
    return str(t)


def _port_info(p, *, is_output: bool = False) -> PortInfo:
    """Shared by introspect_node_class()'s own inputs/outputs below and
    by its new preset resolution -- one real implementation of "how do
    we turn a nodes.core.Port into a PortInfo", not the inputs/outputs
    version copy-pasted a third time for presets."""
    return PortInfo(
        name=p.name,
        type_str=_port_type_str(p.type),
        default=(repr(p.default) if not is_output and not p.required else None),
        required=p.required,
        type_mro=_type_mro(p.type),
        doc=p.doc,
        path_kind=(p.path_kind if not is_output else None),
        choices=(list(p.choices) if p.choices is not None and not is_output else None),
        visible_when=(list(p.visible_when) if p.visible_when is not None and not is_output else None),
        widget_only=(p.widget_only if not is_output else False),
    )


def introspect_node_class(cls: type) -> NodeInfo:
    """Read DECLARED metadata directly off a real nodes.core.Node subclass
    -- INPUTS/OUTPUTS are real Port objects the class author wrote down,
    not guessed from a constructor signature. This is what
    docs/nodes_package_design.md means by "strictly better, since there's
    now an actual contract to read rather than a signature to
    reverse-engineer" -- use this for anything under nodes/, and
    introspect_legacy_class() above for anything not yet migrated there.

    bases is the real Python inheritance chain (excluding object/ABC/the
    dataclass-y ABC noise), so e.g. CAMEOptimizerNode correctly reports
    extending OptimizerNode extending Node -- an actual fact about the
    class, not something inferred or guessed after the fact.

    node_kind/presets are Node.NODE_KIND/list_presets(), verbatim for
    node_kind and resolved-to-PortInfo for presets -- see
    nodes.core.NodePreset's own docstring and
    docs/resources_controller_redesign_plan.md's Phase 3. presets stays
    None for every node_kind == "static" class (the default, true for
    every node in this project today) -- calling list_presets() on a
    static node isn't just unnecessary, nodes.core.Node's own base
    implementation would raise if reached, so this deliberately never
    calls it unless node_kind == "dynamic" says to.
    """
    doc = (inspect.getdoc(cls) or "").strip().split("\n")[0]
    bases = [b.__name__ for b in cls.__mro__[1:] if b.__name__ not in ("object", "ABC")]
    inputs = [_port_info(p) for p in cls.INPUTS.values()]
    outputs = [_port_info(p, is_output=True) for p in cls.OUTPUTS.values()]
    presets = None
    if cls.NODE_KIND == "dynamic":
        presets = [
            PresetInfo(
                name=preset.name,
                required_inputs=[_port_info(p) for p in preset.required_inputs.values()],
                required_outputs=[_port_info(p, is_output=True)
                                   for p in preset.required_outputs.values()],
            )
            for preset in cls.list_presets()
        ]
    from nodes.core import Node  # local: this module stays import-light at module
    # level (see its own docstring) -- cheap and safe here regardless (nodes.core is
    # plain dataclasses/ABC, none of core/__init__.py's ComfyUI-eager-import chain),
    # just kept local to match every other function in this file never doing this.
    # Plain function identity, not .__func__ (diagnostics() is a regular instance
    # method, not a classmethod like list_presets() above -- ClassName.method is
    # already the bare function in Python 3, no bound-classmethod wrapper to unwrap).
    has_diagnostics = cls.diagnostics is not Node.diagnostics
    return NodeInfo(
        class_name=cls.__name__,
        module=cls.__module__,
        doc=doc,
        inputs=inputs,
        outputs=outputs,
        bases=bases,
        display_name=_display_name_for(cls),
        node_kind=cls.NODE_KIND,
        presets=presets,
        has_diagnostics=has_diagnostics,
    )


def introspect_optimizer_nodes() -> list[NodeInfo]:
    """The real thing, superseding introspect_optimizers()'s old
    guess-from-core.optimizers approach: reads declared contracts directly
    off the nodes/optimizer/ package's classes. All five optimizers are
    represented now, including FusedAdafactorOptimizerNode -- which
    correctly shows a FusedOptimizerHandle output type, not just a generic
    OptimizerHandle, because that's what it actually declares (see
    nodes/optimizer/fused_adafactor.py and
    docs/nodes_package_design.md's "fused optimizer family" section).
    """
    from nodes.optimizer.adafactor import AdafactorOptimizerNode
    from nodes.optimizer.came import CAMEOptimizerNode
    from nodes.optimizer.foreach_adafactor import ForeachAdafactorOptimizerNode
    from nodes.optimizer.fused_adafactor import FusedAdafactorOptimizerNode
    from nodes.optimizer.adamw import AdamWOptimizerNode
    return [introspect_node_class(c) for c in (
        AdamWOptimizerNode, AdafactorOptimizerNode, CAMEOptimizerNode,
        ForeachAdafactorOptimizerNode, FusedAdafactorOptimizerNode,
    )]


def node_info_to_dict(info: NodeInfo) -> dict:
    def _ports(ports):
        return [
            {"name": p.name, "type": p.type_str, "default": p.default,
             "required": p.required, "type_mro": p.type_mro,
             "doc": p.doc, "path_kind": p.path_kind, "choices": p.choices,
             "visible_when": p.visible_when, "widget_only": p.widget_only}
            for p in ports
        ]
    return {
        "class_name": info.class_name,
        "display_name": info.display_name,
        "module": info.module,
        "doc": info.doc,
        "bases": info.bases,
        "inputs": _ports(info.inputs),
        "outputs": _ports(info.outputs),
        "node_kind": info.node_kind,
        "presets": (
            [{"name": p.name, "required_inputs": _ports(p.required_inputs),
              "required_outputs": _ports(p.required_outputs)} for p in info.presets]
            if info.presets is not None else None
        ),
        "has_diagnostics": info.has_diagnostics,
    }


def introspect_registry() -> dict[str, list[NodeInfo]]:
    """Every node in server.nodegraph_registry, grouped by domain (derived
    from module path -- see nodegraph_registry.domain_of). This is what the
    interactive graph editor's palette is built from; introspect_optimizer_nodes()
    above predates it and stays only because /nodegraph/optimizers is still
    a valid, narrower endpoint, not because this duplicates it by hand."""
    from . import nodegraph_registry

    groups: dict[str, list[NodeInfo]] = {}
    for cls in nodegraph_registry.get_registry().values():
        domain = nodegraph_registry.domain_of(cls)
        groups.setdefault(domain, []).append(introspect_node_class(cls))
    return groups
