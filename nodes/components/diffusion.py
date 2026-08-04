"""Discrete-time, variance-preserving diffusion process objects (DDPM-style
-- what SDXL actually is). Replaces core.noise_schedule's module-level
ALPHA_T/SIGMA_T globals and core.model_io's free-function I/O convention
with constructed, injectable objects; same math, verified by
smoke_test_diffusion_equivalence.py. A continuous-time process (flow
matching) would be a separate, smaller Interpolant contract, not a subtype
of NoiseSchedule. See docs/training_pipeline_design.md section 1.4.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor


class NoiseSchedule(ABC):

    @abstractmethod
    def alpha_sigma(self, t) -> tuple[Tensor, Tensor]:
        """(alpha, sigma) for timestep index/indices t. Accepts int or
        Tensor, same as core.noise_schedule.get_alpha_sigma."""


class DiscreteLinearNoiseSchedule(NoiseSchedule):
    """Matches ComfyUI's ModelSamplingDiscrete -- same math as
    core.noise_schedule.make_schedule(), moved into an instance instead of
    a module global. Precomputes alpha_t/sigma_t once, on CPU, at
    construction; caches a per-device copy lazily on first use rather than
    mutating a shared global, so two schedules (or the same schedule used
    from two devices in one process) never step on each other."""

    def __init__(self, n: int = 1000, beta_start: float = 0.00085,
                 beta_end: float = 0.012):
        self._alpha_cpu, self._sigma_cpu = self._compute(n, beta_start, beta_end)
        self._device_cache: dict[str, tuple[Tensor, Tensor]] = {}

    @staticmethod
    def _compute(n: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor]:
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n) ** 2
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        alpha_t = alphas_cumprod.sqrt()
        sigma_t = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        return alpha_t, sigma_t

    def alpha_sigma(self, t) -> tuple[Tensor, Tensor]:
        if isinstance(t, Tensor) and t.device.type != "cpu":
            key = str(t.device)
            if key not in self._device_cache:
                self._device_cache[key] = (
                    self._alpha_cpu.to(t.device), self._sigma_cpu.to(t.device))
            a_dev, s_dev = self._device_cache[key]
            return a_dev[t], s_dev[t]
        return self._alpha_cpu[t], self._sigma_cpu[t]


class RescaledZeroTerminalSNRSchedule(DiscreteLinearNoiseSchedule):
    """Same construction as DiscreteLinearNoiseSchedule, but rescales the
    computed alphas_cumprod sequence (Lin et al., "Common Diffusion Noise
    Schedules and Sample Steps are Flawed", arXiv:2305.08891, Sec 3.1) so
    the final value is exactly zero, before deriving alpha_t/sigma_t.
    alphas_cumprod[-1] is exactly 0.0 after this rescale, so sigma_t[-1] is
    exactly inf -- correct by construction, not a bug. Only numerically
    sound paired with VPredParameterization (EpsParameterization.to_x0 is
    degenerate as sigma -> inf); DiffusionProcess.__post_init__ enforces
    that pairing below."""

    @staticmethod
    def _compute(n: int, beta_start: float, beta_end: float) -> tuple[Tensor, Tensor]:
        alpha_t, _ = DiscreteLinearNoiseSchedule._compute(n, beta_start, beta_end)
        sqrt_ac = alpha_t.clone()
        sqrt_ac_T, sqrt_ac_0 = sqrt_ac[-1].clone(), sqrt_ac[0].clone()
        sqrt_ac -= sqrt_ac_T
        sqrt_ac *= sqrt_ac_0 / (sqrt_ac_0 - sqrt_ac_T)
        alphas_cumprod = sqrt_ac ** 2
        return alphas_cumprod.sqrt(), ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5


class Parameterization(ABC):
    """What the model's raw output represents, and how to get x0 (or
    another Parameterization's target) from it. x_t here is in the
    k-diffusion/ModelSamplingDiscrete convention NoiseSchedule uses
    (x_t = x0 + sigma*eps) -- already passed through ModelInputTransform,
    not the raw dataset value."""

    @abstractmethod
    def to_x0(self, raw: Tensor, x_t: Tensor, alpha: Tensor, sigma: Tensor) -> Tensor:
        ...

    @abstractmethod
    def convert_to(self, raw: Tensor, x_t: Tensor, alpha: Tensor, sigma: Tensor,
                    target: "Parameterization") -> Tensor:
        """raw, expressed in this parameterization, converted to what
        target's parameterization would have predicted for the same
        (x_t, alpha, sigma). Same-type conversion is the identity."""


class EpsParameterization(Parameterization):
    """Matches core.noise_schedule.eps_to_x0/eps_to_vpred."""

    def to_x0(self, raw, x_t, alpha, sigma):
        return x_t - sigma * raw

    def convert_to(self, raw, x_t, alpha, sigma, target):
        if isinstance(target, EpsParameterization):
            return raw
        denom_sqrt = torch.sqrt(sigma ** 2 + 1.0)
        return (raw * (sigma ** 2 + 1.0) - x_t * sigma) / denom_sqrt


class VPredParameterization(Parameterization):
    """Matches core.noise_schedule.vpred_to_x0/vpred_to_eps."""

    def to_x0(self, raw, x_t, alpha, sigma):
        denom = sigma ** 2 + 1.0
        return x_t / denom - raw * sigma / torch.sqrt(denom)

    def convert_to(self, raw, x_t, alpha, sigma, target):
        if isinstance(target, VPredParameterization):
            return raw
        denom = sigma ** 2 + 1.0
        return x_t * sigma / denom + raw / torch.sqrt(denom)


class ModelInputTransform(ABC):
    """How a raw x_t (dataset convention) becomes the UNet's actual input."""

    @abstractmethod
    def scale_input(self, x_t: Tensor, sigma) -> Tensor:
        ...


class KarrasInputScaler(ModelInputTransform):
    """xc = x_t / sqrt(sigma^2 + 1) -- matches ComfyUI's calculate_input
    exactly (core.model_io.comfy_input_transform's math, moved into an
    object). Returns bf16 to match the UNet's expected dtype, same as the
    free function it replaces."""

    def scale_input(self, x_t: Tensor, sigma) -> Tensor:
        if isinstance(sigma, Tensor) and sigma.ndim > 0:
            sigma_for_input = (sigma.float() ** 2 + 1.0) ** 0.5
            if sigma_for_input.ndim < 4:
                sigma_for_input = sigma_for_input.view(-1, 1, 1, 1)
            return (x_t / sigma_for_input).to(torch.bfloat16)
        s = sigma if not isinstance(sigma, Tensor) else sigma.item()
        return (x_t / math.sqrt(s ** 2 + 1.0)).to(torch.bfloat16)


@dataclass(frozen=True)
class DiffusionProcess:
    """Everything a training step needs to know about the forward
    diffusion process and the model's I/O convention, as one injected
    dependency -- not three free-function module imports reached for
    individually inside a step loop."""

    schedule: NoiseSchedule
    parameterization: Parameterization
    input_transform: ModelInputTransform

    def __post_init__(self):
        if isinstance(self.schedule, RescaledZeroTerminalSNRSchedule) \
                and isinstance(self.parameterization, EpsParameterization):
            raise ValueError(
                "Zero-terminal-SNR schedules are numerically unsound with "
                "epsilon prediction at t=T (Lin et al. 2023, Sec 3.1) -- "
                "use VPredParameterization.")
