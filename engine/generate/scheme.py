"""How a family's network is sampled: the two parametrisations the families use, behind one small interface.

``EpsScheme`` is a discrete-time noise-prediction model (SDXL): sigmas from a beta table, the network sees a scaled input and an
integer timestep. ``FlowScheme`` is a rectified-flow model (Anima): ``x_s = (1 - s) * x0 + s * noise``, the network is given ``s`` and
predicts the velocity ``noise - x0``, so the clean estimate is ``x - v * s``. A family names its scheme with ``generation``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

from .recipe import Recipe
from .sampling import SAMPLERS, Euler, GuidedDenoiser, Network, NoiseTable, Observed, Peek, SigmaSchedule, initial_noise

ShapeOf = Callable[[Recipe], tuple[int, ...]]  # the latent shape (1, C, h, w) of one recipe's image


class GenerationScheme(Protocol):
    samplers: tuple[str, ...]
    schedulers: tuple[str, ...]

    def check(self, recipe: Recipe) -> None:
        """Raise ``ValueError`` for a setting this model cannot sample with, before anything is loaded."""

    def sample(self, recipes: tuple[Recipe, ...], shape: ShapeOf, network: Network, cfg: float, device: str,
               peek: Peek | None = None) -> Tensor:
        """Latents (one per recipe) on the host. The first recipe's steps, sampler and scheduler apply to all. ``peek(index, estimate)``
        sees the estimate of the clean latents after every network evaluation of the sampler (on the device, not to be kept)."""


class EpsScheme:
    samplers = tuple(SAMPLERS)
    schedulers = tuple(SigmaSchedule.NAMES)

    def check(self, recipe: Recipe) -> None:
        return

    def sample(self, recipes, shape, network, cfg, device, peek=None):
        first, table = recipes[0], NoiseTable()
        sigmas = SigmaSchedule(first.scheduler, table).sigmas(first.steps)
        noise = torch.cat([initial_noise(r.seed, shape(r), float(sigmas[0]), table) for r in recipes]).to(device)
        denoise = GuidedDenoiser(network, table, cfg, first.eps_scale)
        return SAMPLERS[first.sampler]().run(Observed(denoise, peek) if peek else denoise, noise, sigmas).cpu()


class FlowGuidedDenoiser:
    """``denoise(x, sigma)`` for a velocity network: guidance mixes the two velocities, the clean estimate is ``x - v * sigma``."""

    def __init__(self, network: Network, cfg: float):
        self.network, self.cfg = network, cfg

    def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = torch.as_tensor(sigma, dtype=torch.float32)
        t = sigma.expand(x.shape[0]).to(x.device)
        velocity = self.network(x, t, True)
        if self.cfg != 1.0:
            opposite = self.network(x, t, False)
            velocity = opposite + (velocity - opposite) * self.cfg
        return x - velocity.to(x.dtype) * float(sigma)


class FlowScheme:
    """Euler over the shifted flow positions: the sampler ComfyUI runs Anima with.

    ``simple`` and ``normal`` are the same here: evenly spaced positions from 1 to 0, pushed toward noise by the shift. (ComfyUI reads
    ``simple`` off a 1000-entry table of the same curve, so the two agree to within a table step.)
    """

    samplers = ("euler",)
    schedulers = ("simple", "normal")

    def __init__(self, shift: float = 3.0):
        self.shift = shift

    def check(self, recipe: Recipe) -> None:
        if recipe.sampler not in self.samplers:
            raise ValueError(f"this model samples with {', '.join(self.samplers)}, not {recipe.sampler!r}")
        if recipe.scheduler not in self.schedulers:
            raise ValueError(f"this model uses the {' or '.join(self.schedulers)} scheduler, not {recipe.scheduler!r}")
        if recipe.eps_scale != 1.0:
            raise ValueError("epsilon scaling is for noise-prediction models; leave it at 1 for this one")

    def sigmas(self, steps: int) -> Tensor:
        u = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
        return (self.shift * u / (1.0 + (self.shift - 1.0) * u)).float()

    def sample(self, recipes, shape, network, cfg, device, peek=None):
        sigmas = self.sigmas(recipes[0].steps)
        noise = torch.cat([torch.randn(shape(r), generator=torch.Generator().manual_seed(r.seed), dtype=torch.float32) for r in recipes]).to(device)
        denoise = FlowGuidedDenoiser(network, cfg)
        return Euler().run(Observed(denoise, peek) if peek else denoise, noise, sigmas).cpu()


def scheme_for(family) -> GenerationScheme:
    """The scheme a family declares (``generation``); the shift of a flow model is its objective's."""
    kind = family.generation
    if kind == "eps":
        return EpsScheme()
    if kind == "flow":
        return FlowScheme(shift=family.objective().shift)
    raise ValueError(f"family {family.id!r} declares an unknown generation scheme {kind!r}")
