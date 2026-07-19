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


def _type_str(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "any"
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def introspect_class(cls: type, category: str | None = None) -> NodeInfo:
    """Derive a NodeInfo from cls.__init__'s real signature (inputs) and,
    if `category` is given, a single standardized output port (outputs).

    No modification to cls, no instantiation -- read-only introspection.
    Any class works here, not just ones deliberately designed as "nodes" --
    that's intentional for this playground phase: it lets us point this at
    already-existing code (e.g. optimizers.py's classes) and see how close
    the *existing* interface already is to node-shaped, before committing to
    a formal Node/Port base class design.

    Deliberate boundary between what's structurally derived vs. supplied:
    INPUTS are 100% derived from the real signature -- that's a structural
    fact about the class, cannot drift, no second file involved. OUTPUTS
    are different in kind: the standardized rule this codebase is adopting
    is "a node that wraps a constructor has exactly one output: an instance
    of that class" -- but the class itself can't self-report its own
    semantic ROLE in a pipeline (e.g. that ChunkedXPUCAME specifically *is*
    "an Optimizer", as opposed to just being a class with a step() method
    that happens to look like one). That role has to come from the calling
    context that already knows the domain (introspect_optimizers() knows
    it's introspecting optimizers; the class doesn't). So: pass `category`
    explicitly when the caller knows it, and get a real, typed output port
    back. Don't pass it, and outputs comes back empty -- explicitly, rather
    than fabricating a guessed label. A class with no supplied category
    genuinely has no output *yet* in this system; that's honest, not a bug
    to paper over.
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
            required=True,  # "required" isn't really meaningful for an output; kept for a uniform PortInfo shape
        ))
    return NodeInfo(
        class_name=cls.__name__,
        module=cls.__module__,
        doc=doc,
        inputs=ports,
        outputs=outputs,
    )


def introspect_optimizers() -> list[NodeInfo]:
    """First proof-of-concept target: optimizers.py's classes already share
    a real common interface (step(n_steps=), zero_grad(), offload/reload
    hooks) -- see docs/node_architecture_refactor_plan.md Phase 1. Introspect
    them as-is, with zero changes to optimizers.py itself. category="optimizer"
    is supplied here because *this function* knows these classes' pipeline
    role -- see introspect_class()'s docstring for why that can't be derived
    from the classes themselves.
    """
    from core.optimizers import (
        CPUAdamW, ChunkedXPUAdafactor, ChunkedXPUCAME,
        ForeachXPUAdafactor, FusedXPUAdafactor,
    )
    return [introspect_class(c, category="optimizer") for c in (
        CPUAdamW, ChunkedXPUAdafactor, ChunkedXPUCAME,
        ForeachXPUAdafactor, FusedXPUAdafactor,
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
        "inputs": _ports(info.inputs),
        "outputs": _ports(info.outputs),
    }
