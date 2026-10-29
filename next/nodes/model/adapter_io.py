"""Save and load adapter files. Saving is an event subscriber: wire a trigger's ``due`` and the trainer's ``ended`` to it."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

from ...engine.conditioner import Conditioner
from ...engine.handles import TrainableDenoiser
from ...engine.step.save_probe import SaveProbe
from ...families.api.parts import AdapterMerge, DenoiserPart
from ...graph import EventIn, Group, Node, Port
from ...graph.events import CheckpointDue

_SERVICE = "models"


class SaveAdapterNode(Node):
    DISPLAY_NAME = "Save Adapter"
    CATEGORY = "Output"
    INPUTS = {
        "model": Port("model", TrainableDenoiser, doc="Wire the model from Inject Adapter, not the trainer's output: "
                                                       "the saver must exist before training starts."),
        "directory": Port("directory", str, required=False, default="output/adapters"),
        "name": Port("name", str, required=False, default="adapter", doc="File name stem; the step is appended."),
        "format": Port("format", str, required=False, default="auto",
                       doc="File format (key naming scheme). auto: kohya where it can hold the adapter, native for DoRA. "
                           "Choices depend on the model's family: kohya, native."),
        "conditioner": Port("conditioner", Conditioner, required=False, doc="Needed by the probe: the conditioner the trainer uses."),
        "probe": Port("probe", bool, required=False, default=False, group="diagnostics",
                      doc="Before each file is written, run a forward pass with the adapter as it is now and keep its inputs and outputs in "
                          "<file>.probe.safetensors; `accept save-probe` then checks that the saved file reproduces it."),
        "probe_seed": Port("probe_seed", int, required=False, default=0, group="diagnostics", visible_when=("probe", True),
                           doc="Picks the probe prompt and noise (the empty prompt is probed too)."),
    }
    GROUPS = (Group("diagnostics", "Diagnostics", collapsed=True, advanced=True),)
    OUTPUTS = {"saved": Port("saved", int, doc="Files written so far (0 at build time).")}
    EVENTS_IN = {"checkpoint": EventIn(CheckpointDue, handler="on_checkpoint")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        self.model: TrainableDenoiser = inputs["model"]
        requested = inputs.get("format") or "auto"
        if requested == "auto":
            requested = "kohya" if self.model.family.schemes["kohya"].can_write(self.model.kind) else "native"
        scheme = self.model.family.schemes.get(requested)
        if scheme is None:
            raise ValueError(f"Save Adapter: {self.model.family.id} has no {requested!r} format; "
                             f"choose from {sorted(self.model.family.schemes)}")
        if not scheme.can_write(self.model.kind):
            raise ValueError(f"Save Adapter: the {scheme.name} format cannot hold {self.model.kind} weights; pick another format")
        self.scheme, self.directory = scheme, inputs.get("directory") or "output/adapters"
        self.stem, self.written = inputs.get("name") or "adapter", []
        self.probe = None
        if inputs.get("probe"):
            conditioner = inputs.get("conditioner")
            if conditioner is None:
                raise ValueError("Save Adapter: the probe needs the conditioner wired in")
            self.probe = SaveProbe(self.model, getattr(conditioner, "inner", conditioner), seed=inputs.get("probe_seed") or 0)
        return {"saved": 0}

    def _write(self, label: str, step: int) -> str:
        os.makedirs(self.directory, exist_ok=True)
        path = os.path.join(self.directory, f"{self.stem}_{label}.safetensors")
        model = self.model
        if self.probe is not None:  # first: nothing has been exported or written yet, the adapter is exactly what training used
            self.probe.record(path.replace(".safetensors", ".probe.safetensors"), step, os.path.basename(path))
        model.family.save_adapter(path, self.scheme.export(model.adapted), {
            "family": model.family.id, "kind": model.kind, "rank": model.rank, "alpha": model.alpha, "step": step,
            "scheme": self.scheme.name, "base": os.path.basename(model.part.source.path)})
        self.written.append(path)
        return path

    def on_checkpoint(self, event: CheckpointDue, ctx) -> None:
        self._write(f"step{event.step:06d}", event.step)


class LoadAdapterNode(Node):
    DISPLAY_NAME = "Load Adapter"
    CATEGORY = "Model"
    INPUTS = {
        "model": Port("model", TrainableDenoiser),
        "file": Port("file", str, picker="model_file", doc="An adapter under the loras folder."),
        "format": Port("format", str, required=False, default="auto",
                       doc="auto: the format the file names in its metadata, kohya when it names none."),
    }
    OUTPUTS = {"model": Port("model", TrainableDenoiser, doc="The same model with the file's weights loaded.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        model: TrainableDenoiser = inputs["model"]
        path = self.context.env.service(_SERVICE).resolve("loras", inputs["file"])
        state, _metadata = model.family.read_adapter(str(path))
        requested = inputs.get("format") or "auto"
        if requested == "auto":
            requested = (_metadata or {}).get("scheme") or "kohya"
        scheme = model.family.schemes.get(requested)
        if scheme is None:
            raise ValueError(f"Load Adapter: {model.family.id} has no {requested!r} format")
        report = model.family.load_checked(scheme, model.adapted, state)
        model.info = {**model.info, "loaded": {"file": inputs["file"], "matched": report.matched, "missing": list(report.missing)[:20],
                                               "unexpected": list(report.unexpected)[:20]}}
        return {"model": model}


class MergeAdapterNode(Node):
    DISPLAY_NAME = "Merge Adapter"
    CATEGORY = "Model"
    INPUTS = {
        "denoiser": Port("denoiser", DenoiserPart),
        "file": Port("file", str, picker="model_file", doc="A kohya-format adapter under the loras folder."),
        "strength": Port("strength", float, required=False, default=1.0, step=0.05,
                         doc="Multiplies the adapter's own scaling; 1 merges it as saved, negative removes its effect."),
        "format": Port("format", str, required=False, default="kohya", advanced=True),
    }
    OUTPUTS = {"denoiser": Port("denoiser", DenoiserPart, doc="The same denoiser; the adapter is baked in when Inject Adapter loads it. "
                                                              "Chain several nodes to merge several adapters, in order.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: DenoiserPart = inputs["denoiser"]
        merge = AdapterMerge(inputs["file"], float(inputs.get("strength") if inputs.get("strength") is not None else 1.0),
                             inputs.get("format") or "kohya")
        return {"denoiser": replace(part, merges=(*part.merges, merge))}
