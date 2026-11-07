"""Fixture nodes of the new node tree: deterministic shapes, so every validation issue and run outcome the backend
reports has a class that can trigger it, without walking the real node packages. stdlib only: no test using them needs
torch."""

from __future__ import annotations

import time
from typing import Any

from next.graph import Node, Port


class Handle:
    """Marker base class: edge compatibility is a real issubclass check, so a SubHandle satisfies a Handle input."""


class SubHandle(Handle):
    pass


class SumNode(Node):
    INPUTS = {"a": Port("a", float), "b": Port("b", float)}
    OUTPUTS = {"sum": Port("sum", float)}

    def build(self, **inputs):
        return {"sum": inputs["a"] + inputs["b"]}


class ScaleNode(Node):
    INPUTS = {
        "value": Port("value", float),
        "factor": Port("factor", float, required=False, default=2.0),
        "mode": Port("mode", str, required=False, default="mul", choices=("mul", "div")),
    }
    OUTPUTS = {"scaled": Port("scaled", float)}

    def build(self, **inputs):
        value, factor = inputs["value"], inputs["factor"]
        return {"scaled": value * factor if inputs["mode"] == "mul" else value / factor}


class LabelNode(Node):
    INPUTS = {"text": Port("text", str)}
    OUTPUTS = {"label": Port("label", str)}

    def build(self, **inputs):
        return {"label": inputs["text"]}


class ObjectNode(Node):
    """A non-JSON output: it is reported as a ``_type``/``_repr`` summary."""

    INPUTS = {"token": Port("token", str, required=False, default="x")}
    OUTPUTS = {"obj": Port("obj", Any)}

    def build(self, **inputs):
        return {"obj": object()}


class BoomNode(Node):
    """build() always raises: a node failure is a normal outcome."""

    INPUTS = {"trigger": Port("trigger", bool, required=False, default=True)}
    OUTPUTS = {"never": Port("never", bool)}

    def build(self, **inputs):
        raise RuntimeError("boom")


class SlowNode(Node):
    """Sleeps briefly: gives stop and single-active tests a live window."""

    INPUTS = {"seconds": Port("seconds", float, required=False, default=0.3)}
    OUTPUTS = {"slept": Port("slept", float)}

    def build(self, **inputs):
        time.sleep(inputs["seconds"])
        return {"slept": inputs["seconds"]}


class EmitHandleNode(Node):
    INPUTS = {}
    OUTPUTS = {"handle": Port("handle", Handle)}

    def build(self, **inputs):
        return {"handle": Handle()}


class EmitSubHandleNode(Node):
    INPUTS = {}
    OUTPUTS = {"handle": Port("handle", SubHandle)}

    def build(self, **inputs):
        return {"handle": SubHandle()}


class TakeHandleNode(Node):
    INPUTS = {"handle": Port("handle", Handle)}
    OUTPUTS = {"ok": Port("ok", bool)}

    def build(self, **inputs):
        return {"ok": isinstance(inputs["handle"], Handle)}
