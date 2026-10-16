"""Save and load adapter files. Saving is an event subscriber: wire a trigger's ``due`` to it."""

from __future__ import annotations

import os
from typing import Any

from ...engine.handles import TrainableDenoiser
from ...graph import EventIn, Node, Port
from ...graph.events import CheckpointDue, RunStopped

_SERVICE = "models"


class SaveAdapterNode(Node):
    DISPLAY_NAME = "Save Adapter"
    CATEGORY = "Output"
    INPUTS = {
        "model": Port("model", TrainableDenoiser, doc="Wire the model from Inject Adapter, not the trainer's output: "
                                                       "the saver must exist before training starts."),
        "directory": Port("directory", str, required=False, default="output/adapters"),
        "name": Port("name", str, required=False, default="adapter", doc="File name stem; the step is appended."),
        "format": Port("format", str, required=False, default="kohya",
                       doc="File format (key naming scheme). Choices depend on the model's family: kohya, native."),
        "save_final": Port("save_final", bool, required=False, default=True, doc="Also save when the run ends or is stopped."),
    }
    OUTPUTS = {"saved": Port("saved", int, doc="Files written so far (0 at build time).")}
    EVENTS_IN = {"checkpoint": EventIn(CheckpointDue, handler="on_checkpoint"),
                 "stopped": EventIn(RunStopped, handler="on_stopped")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        self.model: TrainableDenoiser = inputs["model"]
        scheme = self.model.family.schemes.get(inputs.get("format") or "kohya")
        if scheme is None:
            raise ValueError(f"Save Adapter: {self.model.family.id} has no {inputs.get('format')!r} format; "
                             f"choose from {sorted(self.model.family.schemes)}")
        if not scheme.can_write(self.model.kind):
            raise ValueError(f"Save Adapter: the {scheme.name} format cannot hold {self.model.kind} weights; pick another format")
        self.scheme, self.directory = scheme, inputs.get("directory") or "output/adapters"
        self.stem, self.save_final, self.written, self.last_step = inputs.get("name") or "adapter", inputs.get("save_final", True), [], 0
        return {"saved": 0}

    def _write(self, label: str, step: int) -> str:
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{self.stem}_{label}.safetensors")
        model = self.model
        model.family.save_adapter(path, self.scheme.export(model.adapted), {
            "family": model.family.id, "kind": model.kind, "rank": model.rank, "alpha": model.alpha, "step": step,
            "scheme": self.scheme.name, "base": os.path.basename(model.part.source.path)})
        self.written.append(path)
        return path

    def on_checkpoint(self, event: CheckpointDue, ctx) -> None:
        self.last_step = event.step
        self._write(f"step{event.step:06d}", event.step)

    def on_stopped(self, event: RunStopped, ctx) -> None:
        if self.save_final and getattr(self, "model", None) is not None and getattr(self.model, "adapted", None) is not None:
            self._write("final", self.last_step)


class LoadAdapterNode(Node):
    DISPLAY_NAME = "Load Adapter"
    CATEGORY = "Model"
    INPUTS = {
        "model": Port("model", TrainableDenoiser),
        "file": Port("file", str, picker="model_file", doc="An adapter under the loras folder."),
        "format": Port("format", str, required=False, default="kohya"),
    }
    OUTPUTS = {"model": Port("model", TrainableDenoiser, doc="The same model with the file's weights loaded.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        model: TrainableDenoiser = inputs["model"]
        scheme = model.family.schemes.get(inputs.get("format") or "kohya")
        if scheme is None:
            raise ValueError(f"Load Adapter: {model.family.id} has no {inputs.get('format')!r} format")
        path = self.context.env.service(_SERVICE).resolve("loras", inputs["file"])
        state, _metadata = model.family.read_adapter(str(path))
        report = model.family.load_checked(scheme, model.adapted, state)
        model.info = {**model.info, "loaded": {"file": inputs["file"], "matched": report.matched, "missing": list(report.missing)[:20],
                                               "unexpected": list(report.unexpected)[:20]}}
        return {"model": model}
