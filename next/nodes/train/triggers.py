"""Triggers: addons that turn what the trainer reports into checkpoint-due events.

They sit on the graph, so one trigger serves every saver in it; a trainer's step events rise to the graph and the savers hear the
event by scope, with no wire (rework/31-node-addons.md, 6c).
"""

from __future__ import annotations

from ...graph import Addon, CheckpointDue, EventIn, EventOut, Port
from ...graph.events import StepFinished


class EveryNSteps(Addon):
    DISPLAY_NAME = "Every N Steps"
    SLOT = "events"
    INPUTS = {"every": Port("every", int, default=250, minimum=1, doc="Emit after every this many optimizer steps.")}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step")}
    EVENTS_OUT = {"due": EventOut(CheckpointDue)}

    def build(self, **inputs) -> EveryNSteps:
        self.every = inputs["every"]
        return self

    def on_step(self, event: StepFinished, ctx) -> None:
        if event.step % self.every == 0:
            self.emit("due", CheckpointDue(event.step, f"every {self.every} steps"))


class EveryEpoch(Addon):
    """In addition to Every N Steps: add either or both."""

    DISPLAY_NAME = "Every Epoch"
    SLOT = "events"
    INPUTS = {"every": Port("every", int, default=1, minimum=1, doc="Emit after every this many epochs.")}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step")}
    EVENTS_OUT = {"due": EventOut(CheckpointDue)}

    def build(self, **inputs) -> EveryEpoch:
        self.every = inputs["every"]
        return self

    def on_step(self, event: StepFinished, ctx) -> None:
        epoch = event.metrics.get("epoch", 0)
        if event.metrics.get("epoch_end") and (epoch + 1) % self.every == 0:
            self.emit("due", CheckpointDue(event.step, f"epoch {epoch + 1}"))
