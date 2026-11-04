"""Addons of the Trainer: optional parts of the training step that need the model and the data stream, which exist only when the trainer is built."""

from __future__ import annotations

from typing import Any

from ...engine.handles import TrainableDenoiser
from ...engine.step.aligned_init import AlignedInit
from ...engine.step.perceptual import SelfPerceptual, SelfPerceptualLoss
from ...engine.step.trend import MODES, TrendTracker
from ...graph import Addon, Port


class ModelBound:
    """What the trainer's ``perceptual`` and ``initializer`` slots hold: a part that is made once the model and the batches are known."""

    def attach(self, model: TrainableDenoiser, batches: Any) -> Any:
        raise NotImplementedError


class _Perceptual(ModelBound):
    def __init__(self, weight: float):
        self.weight = weight

    def attach(self, model: TrainableDenoiser, batches: Any) -> SelfPerceptualLoss:
        objective = batches.noiser.objective
        if objective.process.parameterization.name != "eps":
            raise ValueError("the self-perceptual loss is defined for eps prediction")
        return SelfPerceptualLoss(SelfPerceptual(model.denoiser, objective, model.gate), self.weight)


class _AlignedStart(ModelBound):
    def __init__(self, batches: int):
        self.batches = batches

    def attach(self, model: TrainableDenoiser, batches: Any) -> AlignedInit:
        return AlignedInit(model.adapted.layers, self.batches)


class SelfPerceptualAddon(Addon):
    DISPLAY_NAME = "Self-perceptual loss"
    SLOT = "perceptual"
    INPUTS = {"weight": Port("weight", float, minimum=0.0,
                             doc="The prediction is turned into an image estimate, estimate and true image are noised again at a random timestep, "
                                 "and the frozen base network (the adapter switched off) compares them by its middle-block features. Added to the "
                                 "loss with this weight; the reported loss stays the plain one and the term is reported as loss_perceptual. Needs an "
                                 "adapter with eps prediction, no graph; costs about half a forward and backward more per step.")}

    def build(self, **inputs) -> ModelBound:
        return _Perceptual(inputs["weight"])


class AlignedInitAddon(Addon):
    DISPLAY_NAME = "Gradient-aligned start"
    SLOT = "initializer"
    INPUTS = {"batches": Port("batches", int, default=8, minimum=1,
                              doc="Before the first step the adapter's down-projection factors (A) are set to the main directions of the loss gradient "
                                  "measured on the first N batches (two passes over them, no update; B stays zero, so the model is unchanged). These "
                                  "batches are consumed from the data stream, so training starts N batches in (the region ladder too). "
                                  "LoRA only, a fresh adapter, plain optimizer, no graph.")}

    def build(self, **inputs) -> ModelBound:
        return _AlignedStart(inputs["batches"])


class _Trend(ModelBound):
    def __init__(self, **options: Any):
        self.options = options

    def attach(self, model: TrainableDenoiser, batches: Any) -> TrendTracker:
        return TrendTracker(**self.options)


class TrendWeightingAddon(Addon):
    DISPLAY_NAME = "Trend weighting"
    SLOT = "region_weights"
    INPUTS = {
        "boost": Port("boost", float, default=0.5, minimum=0.0,
                      doc="A region whose part of the timestep range is doing better than usual gets weight 1 + boost * (how much better, in typical successes). "
                          "0 changes nothing. Applied after each region's gradient is scaled to unit length (the `equal` policy), so it changes the direction "
                          "and, by the average weight, the length of the update."),
        "mode": Port("mode", str, required=False, default="trend", choices=MODES,
                     doc="trend: the loss of the region's bins has recently been falling (fast average below slow). current: the region's own loss is "
                         "below the slow average of its bins (noisier: a loss is mostly the draw of image and noise)."),
        "penalty": Port("penalty", float, required=False, default=0.0, minimum=0.0,
                        doc="Lowers the weight (to at least 0.25) of a region doing worse than usual the same way. 0: nothing is taken away."),
        "bins": Port("bins", int, required=False, default=12, minimum=2, advanced=True,
                     doc="Parts the timestep range is cut into; a sample counts for the two nearest in proportion to its closeness."),
        "window": Port("window", int, required=False, default=300, minimum=2, unit="steps", advanced=True, doc="Memory of the slow average."),
        "recent": Port("recent", int, required=False, default=60, minimum=1, unit="steps", advanced=True, doc="Memory of the fast average."),
        "limit": Port("limit", float, required=False, default=3.0, minimum=0.1, advanced=True, doc="Most typical successes that count."),
    }

    def build(self, **inputs) -> ModelBound:
        return _Trend(**{k: inputs[k] for k in ("boost", "mode", "penalty", "bins", "window", "recent", "limit") if k in inputs})
