"""The trainer and its trigger."""

from __future__ import annotations

from typing import Any

from ...data.batcher import Batcher
from ...engine.conditioner import Conditioner
from ...engine.devices import Accelerator
from ...engine.handles import TrainableDenoiser
from ...engine.optim.handle import Optimizer
from ...engine.runtime.density import CheckpointDensity
from ...engine.step.loss import LossWeighting
from ...engine.step.schedule import LRSchedule
from ...engine.step.trainer import Trainer
from ...graph import Completion, EventIn, EventOut, Group, Node, Port
from ...graph.events import CheckpointDue, StepFinished


class DiffusionTrainerNode(Node):
    DISPLAY_NAME = "Diffusion Trainer"
    CATEGORY = "Training"
    READS_SETTINGS = ("device.default",)
    GROUPS = (Group("loop", "Loop"), Group("diagnostics", "Diagnostics", collapsed=True, advanced=True))
    INPUTS = {
        "batches": Port("batches", Batcher),
        "model": Port("model", TrainableDenoiser),
        "conditioner": Port("conditioner", Conditioner),
        "optimizer": Port("optimizer", Optimizer),
        "schedule": Port("schedule", LRSchedule),
        "weighting": Port("weighting", LossWeighting),
        "steps": Port("steps", int, required=False, default=1000, minimum=1, group="loop", doc="Optimizer steps."),
        "accumulation": Port("accumulation", int, required=False, default=1, minimum=1, group="loop",
                             doc="Batches per optimizer step; their gradients average."),
        "start_step": Port("start_step", int, required=False, default=0, minimum=0, group="loop",
                           doc="Step to start from when resuming (the schedule and the logs continue from it)."),
        "tune_checkpointing": Port("tune_checkpointing", bool, required=False, default=False, group="loop",
                                   doc="Before step 1, measure what the card allows and recompute fewer transformer blocks in backward "
                                       "(faster steps) as long as the largest batch still fits the memory budget."),
        "timing": Port("timing", bool, required=False, default=False, group="diagnostics",
                       doc="Measure each phase of the step (synchronizes the device, so it slows training slightly)."),
    }
    OUTPUTS = {"model": Port("model", TrainableDenoiser, doc="The model after training (or after stopping).")}
    EVENTS_OUT = {"step_finished": EventOut(StepFinished, doc="After every optimizer step."),
                  "ended": EventOut(CheckpointDue, doc="Once when training ends, finished or stopped: wire it to a saver for the final file.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        model: TrainableDenoiser = inputs["model"]
        accelerator = Accelerator(model.device)
        residency = getattr(self.context.runtime, "residency", None)
        if residency is not None:
            residency.claim("model", model, rank=10, pinned=True,
                            purpose="frozen weights, adapter factors and whatever else the denoiser keeps")
            residency.claim("optimizer", inputs["optimizer"], rank=1, purpose="optimizer state (moments), one set per adapter factor")
            residency.claim("gradients", inputs["optimizer"].hold_grads(), pinned=True,
                            purpose="adapter gradients, allocated once and zeroed in place so no step allocates between them")
            residency.settle()
            residency.start()
        density = None
        if inputs.get("tune_checkpointing") and residency is not None and model.checkpointer is not None:
            density = (CheckpointDensity(residency.planner), model)
        trainer = Trainer(batches=inputs["batches"], denoiser=model.denoiser, conditioner=inputs["conditioner"],
                          optimizer=inputs["optimizer"], schedule=inputs["schedule"], weighting=inputs["weighting"],
                          steps=inputs.get("steps", 1000), accumulation=inputs.get("accumulation", 1), device=model.device,
                          gate=model.gate, gate_window=model.gate_window,
                          emit=lambda event: self.emit("step_finished", event), should_stop=self.context.should_cancel,
                          synchronize=accelerator.synchronize, timed=bool(inputs.get("timing")), residency=residency, density=density, task=self.context.task)
        start = inputs.get("start_step") or 0
        result = trainer.run(start)
        self.emit("ended", CheckpointDue(start + result.steps_done, "stopped" if result.stopped else "finished"))
        if result.stopped:
            self.report_completion(Completion.unfinished_at(start + result.steps_done, result.total))
        return {"model": model}


class EveryNStepsNode(Node):
    """Turns step events into checkpoint-due events."""

    DISPLAY_NAME = "Every N Steps"
    CATEGORY = "Training"
    INPUTS = {"every": Port("every", int, default=250, minimum=1, doc="Emit after every this many optimizer steps.")}
    OUTPUTS = {"interval": Port("interval", int)}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step")}
    EVENTS_OUT = {"due": EventOut(CheckpointDue)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        self.every = inputs["every"]
        return {"interval": self.every}

    def on_step(self, event: StepFinished, ctx) -> None:
        if event.step % self.every == 0:
            ctx.emit("due", CheckpointDue(event.step, f"every {self.every} steps"))
