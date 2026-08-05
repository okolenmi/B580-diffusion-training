"""LossWeighting: per-sample loss-weight strategies keyed on noise level (sigma)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..components.diffusion import EpsParameterization, Parameterization, VPredParameterization
from ..core import Node, Port


class LossWeighting(ABC):

    @abstractmethod
    def weight(self, sigma: float) -> float:
        ...


class UniformLossWeighting(LossWeighting):

    def weight(self, sigma: float) -> float:
        return 1.0


class MinSNRLossWeighting(LossWeighting):
    """Min-SNR-gamma (Hang et al., "Efficient Diffusion Training via
    Min-SNR Weighting Strategy", ICCV 2023, arXiv:2303.09556). The weight
    factor depends on the student's own prediction parameterization --
    reuses nodes/components/diffusion.py's Parameterization objects for
    that rather than a second, redundant eps/v-pred flag:

    - epsilon (default, matches this class's exact original behavior):
      min(SNR, gamma) / SNR.
    - v-prediction: min(SNR, gamma) / (SNR + 1). Confirmed against the
      paper's own derivation ("when predicting velocity v, the loss
      weight factor must be divided by (SNR+1)") and cross-checked
      against a corrected public reference implementation
      (huggingface/diffusers#5654 -- an earlier version of that same
      example script had this formula wrong, min(SNR+1,gamma)/(SNR+1),
      worth being precise about since it's an easy one to get subtly
      wrong)."""

    def __init__(self, gamma: float = 5.0, parameterization: Parameterization | None = None):
        self.gamma = gamma
        self._parameterization = parameterization or EpsParameterization()

    def weight(self, sigma: float) -> float:
        snr = 1.0 / (sigma ** 2 + 1e-8)
        if isinstance(self._parameterization, VPredParameterization):
            return min(snr, self.gamma) / (snr + 1.0)
        return min(snr, self.gamma) / snr


class P2LossWeighting(LossWeighting):
    """Choi, Lee, Shin, Kim, Kim, Yoon, "Perception Prioritized Training
    of Diffusion Models" (P2 weighting, CVPR 2022, arXiv:2204.00227).
    Weight by 1/(k+SNR)^gamma -- a smoother, more aggressive de-emphasis
    of the highest-SNR (near-clean-image, most imperceptible-detail)
    steps than Min-SNR's hard clamp; worth having as an available option
    rather than assuming Min-SNR is the only reasonable choice."""

    def __init__(self, k: float = 1.0, gamma: float = 1.0):
        self.k, self.gamma = k, gamma

    def weight(self, sigma: float) -> float:
        snr = 1.0 / (sigma ** 2 + 1e-8)
        return 1.0 / ((self.k + snr) ** self.gamma)


class LossWeightingNode(Node):

    OUTPUTS: ClassVar[dict[str, Port]] = {
        "weighting": Port(name="weighting", type=LossWeighting, required=True),
    }

    @abstractmethod
    def build(self, **inputs) -> dict[str, LossWeighting]:
        ...


class UniformLossWeightingNode(LossWeightingNode):

    INPUTS: ClassVar[dict[str, Port]] = {}

    def build(self, **inputs) -> dict[str, LossWeighting]:
        self.validate_inputs(inputs)
        result = {"weighting": UniformLossWeighting()}
        self.validate_outputs(result)
        return result


class MinSNRLossWeightingNode(LossWeightingNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "gamma": Port(name="gamma", type=float, required=False, default=5.0),
        "parameterization": Port(
            name="parameterization", type=Parameterization, required=False, default=None,
            doc="None = EpsParameterization (this class's original, only-correct-until-now "
                "behavior). Wire a VPredParameterization to use the v-prediction form of "
                "Min-SNR instead -- see MinSNRLossWeighting's docstring for both formulas "
                "and how the v-prediction one was verified.",
        ),
    }

    def build(self, **inputs) -> dict[str, LossWeighting]:
        self.validate_inputs(inputs)
        result = {"weighting": MinSNRLossWeighting(
            gamma=inputs.get("gamma", self.INPUTS["gamma"].default),
            parameterization=inputs.get("parameterization"))}
        self.validate_outputs(result)
        return result


class P2LossWeightingNode(LossWeightingNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "k": Port(name="k", type=float, required=False, default=1.0),
        "gamma": Port(name="gamma", type=float, required=False, default=1.0),
    }

    def build(self, **inputs) -> dict[str, LossWeighting]:
        self.validate_inputs(inputs)
        result = {"weighting": P2LossWeighting(
            k=inputs.get("k", self.INPUTS["k"].default),
            gamma=inputs.get("gamma", self.INPUTS["gamma"].default))}
        self.validate_outputs(result)
        return result
