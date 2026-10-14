"""Discrete-time, variance-preserving diffusion (what SDXL is).

Convention (k-diffusion / ComfyUI ModelSamplingDiscrete): the noisy sample is ``x_t = x0 + sigma * eps``
and the network sees ``x_t / sqrt(sigma^2 + 1)``. ``alpha`` is sqrt(alphas_cumprod) and
``sigma = sqrt((1 - alphas_cumprod) / alphas_cumprod)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


class DiscreteSchedule:
    """Scaled-linear betas. Tables are built once on CPU and mirrored per device on first use."""

    def __init__(self, steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        self.steps = steps
        self._alpha_cpu, self._sigma_cpu = self._tables(steps, beta_start, beta_end)
        self._mirrors: dict[str, tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _alphas_cumprod(steps: int, beta_start: float, beta_end: float) -> Tensor:
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps, dtype=torch.float32) ** 2
        return torch.cumprod(1.0 - betas, dim=0)

    @classmethod
    def _tables(cls, steps: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor]:
        acp = cls._alphas_cumprod(steps, beta_start, beta_end)
        return acp.sqrt(), ((1 - acp) / acp).sqrt()

    def alpha_sigma(self, t) -> tuple[Tensor, Tensor]:
        """(alpha, sigma) at integer timestep(s) ``t`` (int or index tensor)."""
        if isinstance(t, Tensor) and t.device.type != "cpu":
            key = str(t.device)
            if key not in self._mirrors:
                self._mirrors[key] = (self._alpha_cpu.to(t.device), self._sigma_cpu.to(t.device))
            alpha, sigma = self._mirrors[key]
            return alpha[t], sigma[t]
        return self._alpha_cpu[t], self._sigma_cpu[t]


class ZeroTerminalSNRSchedule(DiscreteSchedule):
    """Rescaled so the last step is pure noise (Lin et al. 2023, section 3.1). Sigma is infinite at
    the last step by construction, so this pairs only with v-prediction."""

    @classmethod
    def _tables(cls, steps: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor]:
        sqrt_acp = cls._alphas_cumprod(steps, beta_start, beta_end).sqrt()
        last, first = sqrt_acp[-1].clone(), sqrt_acp[0].clone()
        sqrt_acp = (sqrt_acp - last) * (first / (first - last))
        acp = sqrt_acp ** 2
        return acp.sqrt(), ((1 - acp) / acp).sqrt()


class Parameterization:
    """What the raw network output means. ``target`` is what to regress; ``to_x0`` inverts it."""

    name = ""

    def target(self, x0: Tensor, noise: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError

    def to_x0(self, raw: Tensor, x_t: Tensor, sigma: Tensor) -> Tensor:
        raise NotImplementedError

    def to_eps(self, raw: Tensor, x_t: Tensor, sigma: Tensor) -> Tensor:
        return (x_t - self.to_x0(raw, x_t, sigma)) / sigma


class EpsParameterization(Parameterization):
    name = "eps"

    def target(self, x0, noise, sigma):
        return noise

    def to_x0(self, raw, x_t, sigma):
        return x_t - sigma * raw

    def to_eps(self, raw, x_t, sigma):
        return raw


class VPredParameterization(Parameterization):
    name = "v"

    def target(self, x0, noise, sigma):
        return (noise - sigma * x0) / torch.sqrt(sigma ** 2 + 1.0)

    def to_x0(self, raw, x_t, sigma):
        denom = sigma ** 2 + 1.0
        return x_t / denom - raw * sigma / torch.sqrt(denom)

    def to_eps(self, raw, x_t, sigma):
        denom = sigma ** 2 + 1.0
        return x_t * sigma / denom + raw / torch.sqrt(denom)


PARAMETERIZATIONS = {"eps": EpsParameterization, "v": VPredParameterization}


def scale_input(x_t: Tensor, sigma, dtype: torch.dtype | None = None) -> Tensor:
    """The network's input: ``x_t / sqrt(sigma^2 + 1)``, with sigma broadcast over the batch."""
    if isinstance(sigma, Tensor) and sigma.ndim > 0:
        scale = (sigma.float() ** 2 + 1.0).sqrt()
        scale = scale.view(-1, *([1] * (x_t.ndim - 1)))
    else:
        scale = math.sqrt(float(sigma) ** 2 + 1.0)
    out = x_t / scale
    return out if dtype is None else out.to(dtype)


@dataclass(frozen=True)
class DiffusionProcess:
    """A schedule and a parameterization that must travel together."""

    schedule: DiscreteSchedule
    parameterization: Parameterization

    def __post_init__(self) -> None:
        if isinstance(self.schedule, ZeroTerminalSNRSchedule) and isinstance(self.parameterization, EpsParameterization):
            raise ValueError("a zero-terminal-SNR schedule needs v-prediction: eps cannot recover x0 where sigma is infinite")

    def noisy(self, x0: Tensor, noise: Tensor, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """(x_t, target, sigma) for a batch of integer timesteps."""
        _, sigma = self.schedule.alpha_sigma(t)
        sigma = sigma.to(x0.dtype).view(-1, *([1] * (x0.ndim - 1)))
        return x0 + sigma * noise, self.parameterization.target(x0, noise, sigma), sigma
