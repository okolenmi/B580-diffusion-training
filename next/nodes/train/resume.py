"""Resume: saves the training state every N steps and, when the run starts again, continues from the newest one that is whole."""

from __future__ import annotations

from typing import Any

from ...engine.handles import TrainableDenoiser
from ...engine.optim.handle import Optimizer
from ...engine.step.ema import PowerEma
from ...engine.step.resume import ResumeCheckpoint, ResumeStore
from ...graph import EventIn, Node, Port
from ...graph.events import CheckpointDue, StepFinished


class ResumeNode(Node):
    DISPLAY_NAME = "Resume"
    CATEGORY = "Training"
    INPUTS = {
        "model": Port("model", TrainableDenoiser, doc="The model from Inject Adapter."),
        "optimizer": Port("optimizer", Optimizer),
        "ema": Port("ema", PowerEma, required=False, doc="Weight averages from Adapter EMA, kept in the state too."),
        "directory": Port("directory", str, required=False, default="output/resume", doc="Where the states are written (the last two are kept)."),
        "every": Port("every", int, required=False, default=200, minimum=1, unit="steps", doc="Save the state after every this many steps."),
        "identity": Port("identity", str, required=False, default="",
                         doc="Names the run (the starter hashes its settings): a saved state of another run is refused instead of continued. Empty skips the check."),
    }
    OUTPUTS = {"start_step": Port("start_step", int, doc="Wire to the Trainer's `start_step`: the step the saved state was taken after (0 for a fresh run)."),
               "skip": Port("skip", int, doc="Wire to the Batcher's `skip`: the batches already handed out.")}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step"), "checkpoint": EventIn(CheckpointDue, handler="on_end")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        self.every, self.identity = inputs.get("every") or 200, inputs.get("identity") or ""
        self.store = ResumeStore(inputs.get("directory") or "output/resume")
        self.ema = inputs.get("ema")
        self.checkpoint = ResumeCheckpoint(inputs["model"].adapted, inputs["optimizer"], self.ema)
        self._saved = self._position = 0
        state = self.store.latest(say=lambda message: self.context.report("warning", {"node": self.uid, "message": message}))
        if state is None:
            return {"start_step": 0, "skip": 0}
        if self.identity and state.identity != self.identity:
            raise ValueError(f"Resume: {self.store.directory} holds the state of a different run (step {state.step}); delete that folder to start this run from the beginning")
        self.checkpoint.restore(state)
        self._saved, self._position = state.step, state.position
        self.context.report("resume", {"node": self.uid, "message": f"resuming from step {state.step} ({state.position} batches already used)"})
        return {"start_step": state.step, "skip": state.position}

    def on_step(self, event: StepFinished, ctx) -> None:
        self._position = event.position
        if event.step % self.every == 0:
            self._write(event.step)

    def on_end(self, event: CheckpointDue, ctx) -> None:
        """A run stopped on request leaves a state at the step it stopped on. A finished run keeps its states (deleting them here could
        happen before the final adapter file is written); whoever starts runs deletes the folder once the result is safe."""
        if event.reason != "finished" and event.step > self._saved:
            self._write(event.step)

    def _write(self, step: int) -> None:
        if self.ema is not None:
            self.ema.update(step)  # the average may not have seen this step yet
        tensors, meta = self.checkpoint.capture()
        self.store.save(step, self._position, self.identity, tensors, meta)
        self._saved = step
