"""Pure introspection for the node-graph playground (dev/testing tab).

Deliberately mirrors config_schema.py's core principle: UI metadata is
*derived* from the real Python class at import time, never hand-duplicated
in a separate file. That's a structural choice, not a style preference --
the student_mix visibility bug this session existed specifically because
config_ui.py's hand-authored extra_visible_when conditions could (and did)
drift out of sync with what the underlying schema actually supports. A node
graph whose port list comes from inspect.signature() on the real class
cannot drift the same way: change the class, the graph's rendering changes
with it, automatically, with no second file to remember to update.

This module has ZERO side effects and ZERO coupling to the rest of the
codebase beyond importing classes to introspect. It doesn't execute
anything, doesn't touch config, doesn't affect the production training
path in any way. Safe to import, safe to iterate on, safe to delete.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass
class PortInfo:
    name: str
    type_str: str          # best-effort human-readable type (from annotation, or "any" if untyped)
    default: str | None     # repr() of the default, or None if required (no default)
    required: bool


@dataclass
class NodeInfo:
    class_name: str
    module: str
    doc: str                # first line of the class docstring, or ""
    inputs: list[PortInfo]  # derived from __init__'s parameters (minus self)
    outputs: list[PortInfo] # see introspect_class()'s docstring for how these are derived
    bases: list[str] | None = None  # real inheritance chain (Node subclasses only) -- None for legacy-class introspection, where there's no such formal chain to report


def _type_str(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


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
        ))
    outputs = []
    if category is not None:
        outputs.append(PortInfo(
            name=category,
            type_str=cls.__name__,
            default=None,
            required=True,
        ))
    return NodeInfo(
        class_name=cls.__name__,
        module=cls.__module__,
        doc=doc,
        inputs=ports,
        outputs=outputs,
        bases=None,  # no formal Node inheritance chain to report for a guessed class
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
    """
    doc = (inspect.getdoc(cls) or "").strip().split("\n")[0]
    bases = [b.__name__ for b in cls.__mro__[1:] if b.__name__ not in ("object", "ABC")]
    inputs = [
        PortInfo(
            name=p.name,
            type_str=_port_type_str(p.type),
            default=repr(p.default) if not p.required else None,
            required=p.required,
        )
        for p in cls.INPUTS.values()
    ]
    outputs = [
        PortInfo(
            name=p.name,
            type_str=_port_type_str(p.type),
            default=None,
            required=p.required,
        )
        for p in cls.OUTPUTS.values()
    ]
    return NodeInfo(
        class_name=cls.__name__,
        module=cls.__module__,
        doc=doc,
        inputs=inputs,
        outputs=outputs,
        bases=bases,
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
            {"name": p.name, "type": p.type_str, "default": p.default, "required": p.required}
            for p in ports
        ]
    return {
        "class_name": info.class_name,
        "module": info.module,
        "doc": info.doc,
        "bases": info.bases,
        "inputs": _ports(info.inputs),
        "outputs": _ports(info.outputs),
    }
