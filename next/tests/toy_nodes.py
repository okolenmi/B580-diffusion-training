"""Throwaway nodes for exercising the graph core. Not part of the product."""

from __future__ import annotations

import threading

from ..graph import (COMPLETE, Completion, DynamicShape, EpochEnding, EventIn, EventOut, EventType, Group, Node,
                     Port, PromptsChanged, Resolution, StepFinished, StopRequested)


class Table(list):
    """A shared handle: events carry versions, the data lives here."""


class OtherHandle:
    pass


class ConstNode(Node):
    INPUTS = {"value": Port("value", float, required=False, default=1.0)}
    OUTPUTS = {"value": Port("value", float)}

    def build(self, **inputs):
        return {"value": float(inputs["value"])}


class AddNode(Node):
    INPUTS = {"terms": Port("terms", float, required=False, multi=True)}
    OUTPUTS = {"sum": Port("sum", float)}

    def build(self, **inputs):
        self.validate_inputs(inputs)
        return {"sum": sum(inputs["terms"]), "order": list(inputs["terms"])}


class ChoiceNode(Node):
    INPUTS = {"mode": Port("mode", str, choices=("a", "b"), default="a", required=False),
              "n": Port("n", int, required=False, minimum=0, maximum=10)}
    OUTPUTS = {"out": Port("out", str)}

    def build(self, **inputs):
        return {"out": inputs["mode"]}


class TableNode(Node):
    INPUTS = {}
    OUTPUTS = {"table": Port("table", Table)}

    def build(self, **inputs):
        return {"table": Table(["tag-0"])}


class SwapNode(Node):
    """Reactive: rewrites the shared table when an epoch ends, then announces it."""

    INPUTS = {"table": Port("table", Table)}
    OUTPUTS = {"table": Port("table", Table)}
    EVENTS_IN = {"epoch_ending": EventIn(EpochEnding, handler="on_epoch_ending")}
    EVENTS_OUT = {"prompts_changed": EventOut(PromptsChanged)}
    log: list = []

    def build(self, **inputs):
        self.table = inputs["table"]
        self.version = 0
        return {"table": self.table}

    def on_epoch_ending(self, event, ctx):
        self.version += 1
        self.table[:] = [f"tag-{event.epoch + 1}"]
        SwapNode.log.append(("swap", event.epoch))
        ctx.emit("prompts_changed", PromptsChanged(self.version, 1, 1))


class ListenerNode(Node):
    INPUTS = {}
    OUTPUTS = {"ready": Port("ready", bool)}
    EVENTS_IN = {"prompts_changed": EventIn(PromptsChanged, handler="on_changed")}
    log: list = []

    def build(self, **inputs):
        return {"ready": True}

    def on_changed(self, event, ctx):
        ListenerNode.log.append(event.version)


class EpochLoopNode(Node):
    """Stands in for a trainer: finishes epochs, emitting at the boundary, and reads the table after."""

    INPUTS = {"table": Port("table", Table), "epochs": Port("epochs", int, required=False, default=3)}
    OUTPUTS = {"observed": Port("observed", list)}
    EVENTS_OUT = {"epoch_ending": EventOut(EpochEnding), "step": EventOut(StepFinished),
                  "stop": EventOut(StopRequested)}

    def build(self, **inputs):
        observed = []
        for epoch in range(inputs["epochs"]):
            self.emit("epoch_ending", EpochEnding(epoch, 10))
            observed.append(list(inputs["table"]))
        return {"observed": observed}


class FailingNode(Node):
    INPUTS = {}
    OUTPUTS = {"x": Port("x", float)}

    def build(self, **inputs):
        raise RuntimeError("boom")


class TrainerLikeNode(Node):
    """Runs 'steps', stops when cancelled, and says so."""

    INPUTS = {"steps": Port("steps", int, required=False, default=5),
              "cancel_at": Port("cancel_at", int, required=False, default=-1)}
    OUTPUTS = {"model": Port("model", OtherHandle)}
    EVENTS_OUT = {"stop": EventOut(StopRequested), "step": EventOut(StepFinished)}

    def build(self, **inputs):
        done = 0
        for step in range(inputs["steps"]):
            if self.context.should_cancel():
                self.report_completion(Completion.unfinished_at(step=done, total=inputs["steps"]))
                break
            done += 1
            self.emit("step", StepFinished(done, inputs["steps"], {"loss": 0.1}))
            if done == inputs["cancel_at"]:
                self.emit("stop", StopRequested("test"))
        return {"model": OtherHandle()}


class SaverNode(Node):
    INPUTS = {"model": Port("model", OtherHandle)}
    OUTPUTS = {"saved": Port("saved", str)}
    ACCEPTS_UNFINISHED = True

    def build(self, **inputs):
        state = self.upstream_completion("model")
        return {"saved": "partial" if state.unfinished else "full"}


class PassNode(Node):
    INPUTS = {"model": Port("model", OtherHandle)}
    OUTPUTS = {"model": Port("model", OtherHandle)}

    def build(self, **inputs):
        return {"model": inputs["model"]}


class SettingsReaderNode(Node):
    INPUTS = {"undeclared": Port("undeclared", bool, required=False, default=False)}
    OUTPUTS = {"device": Port("device", str)}
    READS_SETTINGS = ("device.default",)

    def build(self, **inputs):
        if inputs["undeclared"]:
            self.context.settings.get("runtime.profile")
        return {"device": self.context.settings.get("device.default")}


class DynamicNode(Node):
    """Output ports follow the 'file' param, like a model loader would."""

    INPUTS = {"file": Port("file", str, required=False, default="")}
    OUTPUTS = {}
    DYNAMIC = DynamicShape(depends_on=("file",))

    @classmethod
    def resolve(cls, params, env=None):
        name = params.get("file") or ""
        if name == "ambiguous":
            return Resolution("needs_choice", dict(cls.INPUTS), {}, choices={"file": ("a", "b")},
                              message="two readings", focus="file")
        if name == "broken":
            return Resolution("error", dict(cls.INPUTS), {}, message="not a model file")
        if name == "raises":
            raise RuntimeError("resolver bug")
        outs = {"vae": Port("vae", OtherHandle)}
        if name == "full":
            outs["denoiser"] = Port("denoiser", Table)
        return Resolution("ready", dict(cls.INPUTS), outs, info=f"{len(outs)} parts")

    def build(self, **inputs):
        res = type(self).resolve({"file": inputs.get("file", "")})
        return {name: (Table() if name == "denoiser" else OtherHandle()) for name in res.outputs}


class GroupedNode(Node):
    GROUPS = (Group("main", "Main"), Group("more", "More", collapsed=True, advanced=True))
    INPUTS = {f"f{i}": Port(f"f{i}", int, required=False, default=0, group="main" if i < 3 else "more")
              for i in range(10)}
    OUTPUTS = {"x": Port("x", int)}

    def build(self, **inputs):
        return {"x": sum(inputs[f"f{i}"] for i in range(10))}
