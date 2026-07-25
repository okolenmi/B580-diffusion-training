"""LossWeighting: per-sample loss-weight strategies keyed on noise level (sigma)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..core import Node, Port


class LossWeighting(ABC):

    @abstractmethod
    def weight(self, sigma: float) -> float:
        ...


class UniformLossWeighting(LossWeighting):

    def weight(self, sigma: float) -> float:
        return 1.0


class MinSNRLossWeighting(LossWeighting):
    """Min-SNR-gamma, epsilon-parameterization form: min(snr, gamma) / snr.
    Only correct for an eps-predicting student -- see
    docs/nodes_package_design.md's TrainerNode section for the
    v-prediction form, not yet implemented here."""

    def __init__(self, gamma: float = 5.0):
        self.gamma = gamma

    def weight(self, sigma: float) -> float:
        snr = 1.0 / (sigma ** 2 + 1e-8)
        return min(snr, self.gamma) / snr


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
    }

    def build(self, **inputs) -> dict[str, LossWeighting]:
        self.validate_inputs(inputs)
        result = {"weighting": MinSNRLossWeighting(
            gamma=inputs.get("gamma", self.INPUTS["gamma"].default))}
        self.validate_outputs(result)
        return result
