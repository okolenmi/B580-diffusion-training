"""Turning noise into an image: sigma schedules, samplers and guidance for a discrete-time eps model.

The conventions are those of ComfyUI (k-diffusion): the sample is ``x = x0 + sigma * eps``, the network sees
``x / sqrt(sigma^2 + 1)`` and the nearest integer timestep of sigma, and guidance mixes the predictions of the two prompts.
The same seed, steps, sampler and scheduler therefore start from the same noise and take the same path as there.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor


class NoiseTable:
    """The model's sigma at each of its integer timesteps, and the conversions between the two."""

    def __init__(self, steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.012):
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, steps, dtype=torch.float64) ** 2  # float64 as the model's published tables are
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.sigmas = (((1 - alphas_cumprod) / alphas_cumprod) ** 0.5).float()
        self.log_sigmas = self.sigmas.log()
        self.sigma_min, self.sigma_max = float(self.sigmas[0]), float(self.sigmas[-1])

    def timestep(self, sigma: float | Tensor) -> Tensor:
        """The integer timestep whose sigma is nearest in log (what the network is given)."""
        log_sigma = torch.as_tensor(sigma, dtype=torch.float32).log()
        return (log_sigma - self.log_sigmas[:, None]).abs().argmin(dim=0).view(log_sigma.shape)

    def sigma(self, timestep: Tensor) -> Tensor:
        """Sigma at a fractional timestep, interpolated in log."""
        t = timestep.float().clamp(0, len(self.sigmas) - 1)
        low, high, weight = t.floor().long(), t.ceil().long(), t.frac()
        return ((1 - weight) * self.log_sigmas[low] + weight * self.log_sigmas[high]).exp()


class SigmaSchedule:
    """The sigmas a run walks through, from the largest to 0 (``steps + 1`` values)."""

    def __init__(self, name: str, table: NoiseTable):
        if name not in self.NAMES:
            raise ValueError(f"unknown scheduler {name!r}; choose from {tuple(self.NAMES)}")
        self.name, self.table = name, table

    def sigmas(self, steps: int) -> Tensor:
        return getattr(self, f"_{self.name}")(steps)

    def _normal(self, steps: int) -> Tensor:
        table = self.table
        timesteps = torch.linspace(float(table.timestep(table.sigma_max)), float(table.timestep(table.sigma_min)), steps)
        return torch.cat([table.sigma(timesteps), torch.zeros(1)])

    def _simple(self, steps: int) -> Tensor:
        table, stride = self.table, len(self.table.sigmas) / steps
        picked = [float(table.sigmas[-(1 + int(i * stride))]) for i in range(steps)]
        return torch.tensor(picked + [0.0])

    def _karras(self, steps: int, rho: float = 7.0) -> Tensor:
        ramp = torch.linspace(0, 1, steps)
        high, low = self.table.sigma_max ** (1 / rho), self.table.sigma_min ** (1 / rho)
        return torch.cat([(high + ramp * (low - high)) ** rho, torch.zeros(1)])

    NAMES = {"normal": _normal, "simple": _simple, "karras": _karras}


Denoise = Callable[[Tensor, Tensor], Tensor]  # (x, sigma) -> the estimate of the clean sample


class Sampler:
    name = ""

    def run(self, denoise: Denoise, x: Tensor, sigmas: Tensor) -> Tensor:
        raise NotImplementedError


class Euler(Sampler):
    name = "euler"

    def run(self, denoise: Denoise, x: Tensor, sigmas: Tensor) -> Tensor:
        for i in range(len(sigmas) - 1):
            denoised = denoise(x, sigmas[i])
            x = x + (x - denoised) / sigmas[i] * (sigmas[i + 1] - sigmas[i])
        return x


class DPMPP2M(Sampler):
    """DPM-Solver++ (2M): a second-order multistep solver in log-sigma."""

    name = "dpmpp_2m"

    def run(self, denoise: Denoise, x: Tensor, sigmas: Tensor) -> Tensor:
        previous = None
        for i in range(len(sigmas) - 1):
            denoised = denoise(x, sigmas[i])
            t, t_next = sigmas[i].log().neg(), sigmas[i + 1].log().neg()
            h = t_next - t
            if previous is None or sigmas[i + 1] == 0:
                used = denoised
            else:
                r = (t - sigmas[i - 1].log().neg()) / h
                used = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * previous
            x = (t_next.neg().exp() / t.neg().exp()) * x - (-h).expm1() * used
            previous = denoised
        return x


SAMPLERS = {cls.name: cls for cls in (Euler, DPMPP2M)}


class Network(Protocol):
    def __call__(self, x_in: Tensor, t: Tensor, positive: bool) -> Tensor: ...


class GuidedDenoiser:
    """``denoise(x, sigma)`` for a network that predicts eps for a positive and a negative prompt (classifier-free guidance)."""

    def __init__(self, network: Network, table: NoiseTable, cfg: float, eps_scale: float = 1.0):
        self.network, self.table, self.cfg, self.eps_scale = network, table, cfg, eps_scale

    def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        t = self.table.timestep(sigma).float().expand(x.shape[0]).to(x.device)
        x_in = x / math.sqrt(float(sigma) ** 2 + 1.0)
        eps = self.network(x_in, t, True)
        if self.cfg != 1.0:
            eps_negative = self.network(x_in, t, False)
            eps = eps_negative + (eps - eps_negative) * self.cfg
        if self.eps_scale != 1.0:
            eps = eps / self.eps_scale  # epsilon scaling: the guided prediction is shrunk a little, as sampling errors make it too large
        return x - eps.to(x.dtype) * float(sigma)


def initial_noise(seed: int, shape: tuple[int, ...], first_sigma: float, table: NoiseTable) -> Tensor:
    """The seed's noise, scaled the way a run that starts from the largest sigma does."""
    noise = torch.randn(shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    if math.isclose(first_sigma, table.sigma_max, rel_tol=1e-5) or first_sigma > table.sigma_max:
        return noise * math.sqrt(1.0 + first_sigma ** 2)
    return noise * first_sigma
