"""Loss weighting: how much each sample counts, by its noise level."""

from __future__ import annotations

from typing import Any

from ...engine.step.balance import MODES, TimeBalance
from ...engine.step.loss import WEIGHTINGS, LossWeighting
from ...graph import Group, Node, Port


class LossWeightingNode(Node):
    DISPLAY_NAME = "Loss Weighting"
    CATEGORY = "Training"
    INPUTS = {
        "kind": Port("kind", str, required=False, default="uniform", choices=tuple(WEIGHTINGS)),
        "parameterization": Port("parameterization", str, required=False, default="eps", choices=("eps", "v"),
                                 visible_when=("kind", "min_snr"), doc="What the network predicts; Min-SNR differs for v."),
        "gamma": Port("gamma", float, required=False, default=5.0, minimum=0.0, visible_when=("kind", "min_snr")),
        "k": Port("k", float, required=False, default=1.0, visible_when=("kind", "p2")),
        "p2_gamma": Port("p2_gamma", float, required=False, default=1.0, visible_when=("kind", "p2")),
    }
    OUTPUTS = {"weighting": Port("weighting", LossWeighting)}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        kind = inputs.get("kind") or "uniform"
        cls = WEIGHTINGS[kind]
        if kind == "min_snr":
            return {"weighting": cls(inputs.get("gamma", 5.0), inputs.get("parameterization") or "eps")}
        if kind == "p2":
            return {"weighting": cls(inputs.get("k", 1.0), inputs.get("p2_gamma", 1.0))}
        return {"weighting": cls()}


class TimeBalanceNode(Node):
    DISPLAY_NAME = "Time Balance"
    CATEGORY = "Training"
    GROUPS = (Group("weights", "Loss weights"), Group("sampling", "Sampling"), Group("tracking", "Tracking", collapsed=True, advanced=True))
    INPUTS = {
        "mode": Port("mode", str, required=False, default="speed", choices=MODES, group="weights",
                     doc="off: only track. normalize: equal loss scale per third. speed: thirds that improve slower gain weight. "
                         "dro: the third furthest above its own baseline dominates."),
        "eta": Port("eta", float, required=False, default=0.5, minimum=0.0, group="weights", visible_when=("mode", "speed")),
        "leak": Port("leak", float, required=False, default=0.05, minimum=0.0, maximum=0.99, group="weights", visible_when=("mode", "speed"),
                     doc="Share of the accumulated weight forgotten at every report, so a lasting weight needs a lasting difference in speed."),
        "dro_lambda": Port("dro_lambda", float, required=False, default=1.0, minimum=0.0, group="weights", visible_when=("mode", "dro")),
        "clip_min": Port("clip_min", float, required=False, default=0.25, minimum=0.0, group="weights"),
        "clip_max": Port("clip_max", float, required=False, default=4.0, minimum=0.0, group="weights"),
        "sample_bias": Port("sample_bias", float, required=False, default=0.0, minimum=0.0, group="sampling",
                            doc="Draw timesteps more often from the thirds that are behind: chance proportional to (loss / baseline) to this power. "
                                "0 leaves the Noiser's own distribution; wire this node to the Noiser to use it."),
        "warmup_reports": Port("warmup_reports", int, required=False, default=10, minimum=1, group="tracking",
                               doc="Steps a third is observed before its baseline is fixed and it can be weighted."),
        "ema_alpha": Port("ema_alpha", float, required=False, default=0.05, minimum=0.0, maximum=1.0, group="tracking"),
        "fast_alpha": Port("fast_alpha", float, required=False, default=0.25, minimum=0.0, maximum=1.0, group="tracking"),
    }
    OUTPUTS = {"balance": Port("balance", TimeBalance, doc="Wire to the Trainer (weights) and, for sampling, to the Noiser.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        def get(name, default):
            return default if inputs.get(name) is None else inputs[name]
        return {"balance": TimeBalance(get("mode", "speed"), get("warmup_reports", 10), get("ema_alpha", 0.05), get("fast_alpha", 0.25),
                                       get("eta", 0.5), get("clip_min", 0.25), get("clip_max", 4.0), get("dro_lambda", 1.0),
                                       get("sample_bias", 0.0), get("leak", 0.05))}
