"""Fixture nodes of the new node tree: deterministic shapes, so every validation issue and run outcome the backend
reports has a class that can trigger it, without walking the real node packages. stdlib only: no test using them needs
torch."""

from __future__ import annotations

import time
from typing import Any

from graph import ActionModule, CheckpointDue, DisplayModule, EventIn, EventOut, Group, Node, Port, SaveRequested, Series


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


class DiagnosingNode(Node):
    INPUTS = {"path": Port("path", str, required=False, default="")}
    OUTPUTS = {"y": Port("y", str)}

    def diagnostics(self, inputs):
        if inputs.get("path") == "unreadable":
            raise ValueError("path does not resolve")
        return {"path": [f"looked at {inputs.get('path')!r}"]}

    def build(self, **inputs):
        return {"y": inputs["path"]}


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


class WidgetNode(Node):
    """A node with live views: reports its loss on its own stream, as a node with MODULES does."""

    GROUPS = (Group("run", "Run"),)
    INPUTS = {
        "steps": Port("steps", int, required=False, default=3, label="Steps", group="run", unit="steps",
                      minimum=1, maximum=100, step=1),
        "tag": Port("tag", str, required=False, default="x", advanced=True),
    }
    OUTPUTS = {"done": Port("done", int)}
    MODULES = (
        DisplayModule(name="loss", kind="chart", title="Loss", scale="symlog", series=(Series("loss", "loss", "#6c8cff"),)),
        DisplayModule(name="progress", kind="progress", title="Progress", value_key="step", total_key="total_steps"),
    )

    def build(self, **inputs):
        steps = inputs["steps"]
        for i in range(steps):
            self.context.report(f"node:{self.uid}", {"step": i + 1, "total_steps": steps, "loss": 1.0 / (i + 1)})
        return {"done": steps}


#: What CheckpointListenerNode heard in this process (the in-process gateway runs the graph here).
HEARD: list = []


class ActionWaitNode(Node):
    """Works until a page's "save now" reaches it (or the run is cancelled), handing posted commands over as a trainer does."""

    OUTPUTS = {"done": Port("done", int)}
    EVENTS_IN = {"save_now": EventIn(SaveRequested, handler="on_save_now")}
    EVENTS_OUT = {"checkpoint": EventOut(CheckpointDue)}
    ACTIONS = (ActionModule("save_now", "Save now", "save_now", needs=CheckpointDue, needs_label="a listener"),)

    def build(self, **inputs):
        self.saved = False
        self.listen_now()
        deadline = time.monotonic() + 20
        while not self.saved and not self.context.should_cancel() and time.monotonic() < deadline:
            self.context.events.drain()
            time.sleep(0.02)
        return {"done": int(self.saved)}

    def on_save_now(self, event, ctx):
        self.saved = True
        self.emit("checkpoint", CheckpointDue(7, event.reason))


class CheckpointListenerNode(Node):
    OUTPUTS = {"ok": Port("ok", int)}
    EVENTS_IN = {"checkpoint": EventIn(CheckpointDue, handler="on_checkpoint", scoped=True)}

    def build(self, **inputs):
        HEARD.clear()
        return {"ok": 1}

    def on_checkpoint(self, event, ctx):
        HEARD.append((event.step, event.reason))
