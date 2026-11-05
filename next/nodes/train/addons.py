"""Addons of the Trainer: optional parts of the training step that need the model and the data stream, which exist only when the trainer is built."""

from __future__ import annotations

from typing import Any

from ...engine.handles import TrainableDenoiser
from ...engine.step.aligned_init import AlignedInit
from ...engine.step.anchor import PROMPTS, AnchorLoss
from ...engine.step.perceptual import SelfPerceptual, SelfPerceptualLoss
from ...engine.step.policy import make_policy
from ...engine.step.regional import RegionalUpdate
from ...engine.step.trend import MODES, TrendTracker
from ...graph import Addon, Port


class ModelBound:
    """What the trainer's ``perceptual`` and ``initializer`` slots hold: a part that is made once the model and the batches are known."""

    def attach(self, model: TrainableDenoiser, batches: Any) -> Any:
        raise NotImplementedError


class ConditionerBound:
    """What the trainer's ``anchor`` slot holds: a part that is made once the model and the conditioner are known."""

    def attach(self, model: TrainableDenoiser, conditioner: Any, accumulation: int) -> Any:
        raise NotImplementedError


class _Anchor(ConditionerBound):
    def __init__(self, weight: float, every: int, prompts: tuple[str, ...]):
        self.weight, self.every, self.prompts = weight, every, prompts

    def attach(self, model: TrainableDenoiser, conditioner: Any, accumulation: int) -> AnchorLoss:
        return AnchorLoss(model.denoiser, conditioner, model.gate, self.prompts, self.weight, self.every, accumulation)


class AnchorAddon(Addon):
    DISPLAY_NAME = "Anchor loss"
    SLOT = "anchor"
    INPUTS = {
        "weight": Port("weight", float, default=1.0, minimum=0.0,
                       doc="Every few steps the network is also asked about the batch's noisy images under a random prompt unrelated to the dataset, with and "
                           "without the adapter; the squared difference of the two predictions, times this weight, is added to the gradient. It keeps the adapter "
                           "from changing the model's behaviour for other prompts. Reported as loss_anchor (before the weight). Plain optimizer, no graph; "
                           "an anchor step costs about one more forward and backward."),
        "every": Port("every", int, required=False, default=2, minimum=1, unit="steps", doc="Steps between anchor passes (with regions, every region's pass of that step runs one)."),
        "prompts": Port("prompts", str, required=False, default="",
                        doc="The anchor prompts separated by `|`; empty uses a built-in list of eight unrelated scenes and portraits."),
    }

    def build(self, **inputs) -> ConditionerBound:
        text = (inputs.get("prompts") or "").strip()
        prompts = tuple(p.strip() for p in text.split("|") if p.strip()) if text else PROMPTS
        return _Anchor(inputs["weight"], inputs.get("every") or 2, prompts)


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


class RegionPolicy(ModelBound):
    """What the trainer's ``gradient_policy`` slot holds: how the regions of a batch are reconciled, and how often they run alone."""

    def calibrate_every(self, batches: Any) -> int:
        raise NotImplementedError


class _Policy(RegionPolicy):
    def __init__(self, kind: str, normalize: bool, floor: float, calibrate_every: int):
        self.kind, self.normalize, self.floor, self._calibrate_every = kind, normalize, floor, calibrate_every

    def attach(self, model: TrainableDenoiser, batches: Any) -> RegionalUpdate:
        if batches.noiser.max_regions < 2:
            raise ValueError(f"gradient policy {self.kind!r} reconciles regions: set the noiser's regions to 2 or more")
        policy = make_policy(self.kind, self.floor, int(getattr(batches, "seed", 0)))
        return RegionalUpdate(policy, self.normalize or self.kind == "equal")

    def calibrate_every(self, batches: Any) -> int:
        noiser = batches.noiser
        if self._calibrate_every and (self.kind != "equal" or noiser.max_regions != 3 or noiser.policy.layout != "strata"):
            raise ValueError("calibrating needs gradient policy equal and the noiser's regions 3 with the strata layout")
        return self._calibrate_every


class GradientPolicyAddon(Addon):
    DISPLAY_NAME = "Gradient policy"
    SLOT = "gradient_policy"
    INPUTS = {
        "policy": Port("policy", str, default="equal", choices=("equal", "project", "common"),
                     doc="How the regions of a batch (see the noiser's `regions`) are reconciled. equal: each region's gradient is scaled to unit length, "
                         "then averaged, so every region counts the same whatever its loss weight. project: a region's gradient loses the part that points "
                         "against another region's. common: the shortest vector among the regions' gradients, which lowers every region's loss to first order. "
                         "The last two run each region's backward alone (more memory-light but slower) and need the plain optimizer, no graph. "
                         "Without this addon the batch is one loss (the plain mean)."),
        "normalize": Port("normalize", bool, required=False, default=True,
                          doc="Scale each region's gradient to unit length before combining, so heavily down-weighted regions count as much as the rest."),
        "floor": Port("floor", float, required=False, default=0.0, minimum=0.0, maximum=1.0, visible_when=("policy", "common"),
                      doc="Skip the update when the shared direction keeps less than this share of the average squared gradient norm "
                          "(1 = regions identical, 1 / regions = unrelated, 0 = they cancel). 0 never skips."),
        "calibrate_every": Port("calibrate_every", int, required=False, default=0, minimum=0, visible_when=("policy", "equal"),
                                doc="0: every step runs each region alone (one forward and backward per region). N: only every Nth step does, and measures each "
                                    "region's gradient norm; the steps between run the regions as one batch (one forward and backward), each region's loss weighted "
                                    "by the inverse of its measured norm. Needs `--time-balance gradnorm`, 3 regions with the `strata` layout."),
    }

    def build(self, **inputs) -> RegionPolicy:
        return _Policy(inputs["policy"], bool(inputs.get("normalize", True)), float(inputs.get("floor") or 0.0), int(inputs.get("calibrate_every") or 0))
