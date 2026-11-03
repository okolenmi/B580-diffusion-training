"""Adapter EMA: power-function moving averages of the adapter weights, kept while training and written by Save Adapter."""

from __future__ import annotations

from typing import Any

from ...engine.handles import TrainableDenoiser
from ...engine.step.ema import PowerEma
from ...graph import EventIn, Node, Port
from ...graph.events import StepFinished


class AdapterEmaNode(Node):
    DISPLAY_NAME = "Adapter EMA"
    CATEGORY = "Training"
    INPUTS = {
        "model": Port("model", TrainableDenoiser, doc="The model from Inject Adapter."),
        "sigma_rel": Port("sigma_rel", str, required=False, default="0.05,0.1",
                          doc="Window widths, comma-separated, as a fraction of the run so far (0.05: about the last 5-10% of the steps). "
                              "One average is kept per width and Save Adapter writes a file for each; costs one extra copy of the adapter on the card per width."),
    }
    OUTPUTS = {"ema": Port("ema", PowerEma, doc="Wire to Save Adapter's `ema`.")}
    EVENTS_IN = {"step": EventIn(StepFinished, handler="on_step")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        widths = [float(v) for v in (inputs.get("sigma_rel") or "").split(",") if v.strip()]
        self.ema = PowerEma(inputs["model"].adapted.trainable_parameters(), widths)
        return {"ema": self.ema}

    def on_step(self, event: StepFinished, ctx) -> None:
        self.ema.update(event.step)
