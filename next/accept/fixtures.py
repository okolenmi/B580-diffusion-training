"""Fixed, repeatable inputs for running a denoiser the same way twice."""

from __future__ import annotations

import torch

from ..families.sdxl.objective import SDXLObjective


class DenoiserInputs:
    """Noisy latents at chosen timesteps with random text conditioning (``context`` tokens of ``context_dim``, pooled ``adm`` vector), all from one seed.
    ``run(denoiser)`` runs forward and backward against a fixed output gradient and returns ``(output, input_gradient)``."""

    def __init__(self, *, resolution: int, timesteps: list[int], device, dtype, context_dim: int = 2048, adm_dim: int = 2816, tokens: int = 77, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        batch, latent = len(timesteps), resolution // 8
        x0 = torch.randn(batch, 4, latent, latent, generator=gen)
        t = torch.tensor(timesteps)
        x_t, _, _ = SDXLObjective().noisy(x0, torch.randn(x0.shape, generator=gen), t)
        self.x = x_t.to(device, dtype)
        self.t = t.to(device)
        self.context = torch.randn(batch, tokens, context_dim, generator=gen).to(device, dtype)
        self.adm = torch.randn(batch, adm_dim, generator=gen).to(device, dtype)
        self.dy = torch.randn(x0.shape, generator=gen).to(device, dtype)

    def run(self, denoiser) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.x.clone().requires_grad_(True)
        with torch.enable_grad():
            out = denoiser(x, self.t, self.context, self.adm)
            out.backward(self.dy)
        return out, x.grad
