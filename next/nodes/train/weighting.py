"""Loss weighting: how much each sample counts, by its noise level."""

from __future__ import annotations

from typing import Any

from ...engine.step.loss import WEIGHTINGS, LossWeighting
from ...graph import Node, Port


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
