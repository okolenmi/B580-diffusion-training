"""Throwaway hosts and addons for exercising addons and event scopes. Not part of the product."""

from __future__ import annotations

from ..graph import (Addon, AddonSlot, CheckpointDue, EventIn, EventOut, Node, Port, StepFinished)

LOG: list[tuple] = []


class Factor:
    """The interface of the ``factor`` and ``offset`` slots."""

    def apply(self, x: float) -> float:
        raise NotImplementedError


class Times(Factor):
    def __init__(self, by: float):
        self.by = by

    def apply(self, x: float) -> float:
        return x * self.by


class Plus(Factor):
    def __init__(self, by: float):
        self.by = by

    def apply(self, x: float) -> float:
        return x + self.by


class FactorNode(Node):
    INPUTS = {"by": Port("by", float, required=False, default=10.0)}
    OUTPUTS = {"factor": Port("factor", Factor)}

    def build(self, **inputs):
        return {"factor": Times(inputs["by"])}


class Calc(Node):
    INPUTS = {"x": Port("x", float, required=False, default=1.0)}
    OUTPUTS = {"y": Port("y", float)}
    ADDON_SLOTS = (AddonSlot("factor", Factor), AddonSlot("offset", Factor, many=True))

    def build(self, **inputs):
        y = inputs["x"]
        factor = self.addons.one("factor")
        y = factor.apply(y) if factor else y
        for offset in self.addons.all("offset"):
            y = offset.apply(y)
        return {"y": y}


class CalcTwo(Calc):
    """Same slots as Calc under another class name, for FITS."""


class Number(Node):
    INPUTS = {"value": Port("value", float, required=False, default=1.0)}
    OUTPUTS = {"value": Port("value", float)}

    def build(self, **inputs):
        return {"value": inputs["value"]}


class OtherCalc(Node):
    """A second host class with the same slot: the addons work here without a change."""

    INPUTS = {"x": Port("x", float, required=False, default=1.0)}
    OUTPUTS = {"y": Port("y", float)}
    ADDON_SLOTS = (AddonSlot("factor", Factor),)

    def build(self, **inputs):
        factor = self.addons.one("factor")
        return {"y": factor.apply(inputs["x"]) if factor else inputs["x"]}


class Plain(Node):
    OUTPUTS = {"out": Port("out", int)}

    def build(self, **inputs):
        return {"out": 1}


class Scale(Addon):
    SLOT = "factor"
    INPUTS = {"by": Port("by", float, required=False, default=2.0, minimum=0.0)}

    def build(self, **inputs):
        return Times(inputs["by"])


class Shift(Addon):
    SLOT = "offset"
    INPUTS = {"by": Port("by", float, required=False, default=1.0)}

    def build(self, **inputs):
        return Plus(inputs["by"])


class NeedsScale(Addon):
    SLOT = "offset"
    REQUIRES = ("Scale",)

    def build(self, **inputs):
        return Plus(0.0)


class NoShift(Addon):
    SLOT = "offset"
    CONFLICTS = ("Shift",)

    def build(self, **inputs):
        return Plus(0.0)


class OnlyCalc(Addon):
    SLOT = "offset"
    FITS = ("Calc",)

    def build(self, **inputs):
        return Plus(0.0)


class Stepper(Node):
    """Emits ``steps`` StepFinished events while it builds."""

    INPUTS = {"steps": Port("steps", int, required=False, default=5)}
    OUTPUTS = {"done": Port("done", int)}
    EVENTS_OUT = {"step": EventOut(StepFinished)}

    def build(self, **inputs):
        for i in range(1, inputs["steps"] + 1):
            self.emit("step", StepFinished(i, inputs["steps"], {}))
        return {"done": inputs["steps"]}


class Sink(Node):
    """Hears CheckpointDue by scope alone, so no wire is needed."""

    OUTPUTS = {"seen": Port("seen", int)}
    EVENTS_IN = {"due": EventIn(CheckpointDue, "on_due", scoped=True)}

    def build(self, **inputs):
        return {"seen": 0}

    def on_due(self, event, context):
        LOG.append(("sink", event.step))


class Trigger(Addon):
    """Every N steps of what it hears, emit CheckpointDue."""

    SLOT = "events"
    INPUTS = {"every": Port("every", int, required=False, default=2, minimum=1)}
    EVENTS_IN = {"step": EventIn(StepFinished, "on_step")}
    EVENTS_OUT = {"due": EventOut(CheckpointDue)}

    def build(self, **inputs):
        self.every = inputs["every"]
        return self

    def on_step(self, event, context):
        if event.step % self.every == 0:
            self.emit("due", CheckpointDue(event.step, f"every {self.every}"))


class Count(Addon):
    SLOT = "events"
    EVENTS_IN = {"step": EventIn(StepFinished, "on_step")}

    def on_step(self, event, context):
        LOG.append(("count", self.address, event.step))


class Stop(Addon):
    SLOT = "events"
    EVENTS_IN = {"step": EventIn(StepFinished, "on_step")}

    def on_step(self, event, context):
        context.stop()
