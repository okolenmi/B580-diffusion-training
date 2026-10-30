"""Loss at fixed timesteps with and without the adapter, on images the adapter was trained on and on images it never saw.

The training loss mixes whatever timesteps the steps happened to draw. This measures every timestep of a fixed grid on the same noise,
so a region that got worse is visible, and the held-out images say whether an improvement is learning or memorising.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..families.sdxl.objective import SDXLObjective

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_T = (25, 75, 150, 250, 350, 450, 550, 650, 750, 850, 950)


def load_images(folder: str, count: int, size: int) -> list[Tensor]:
    """``count`` images spread evenly over the sorted files, each cropped to a centred square and resized: (3, size, size) uint8."""
    from PIL import Image

    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_TYPES)
    if not files:
        raise SystemExit(f"no images in {folder}")
    picks = [files[int(i * len(files) / min(count, len(files)))] for i in range(min(count, len(files)))]
    out = []
    for path in picks:
        with Image.open(path) as image:
            image = image.convert("RGB")
            side = min(image.size)
            left, top = (image.width - side) // 2, (image.height - side) // 2
            out.append(torch.from_numpy(np.array(image.crop((left, top, left + side, top + side)).resize((size, size)))).permute(2, 0, 1).clone())
    return out


class TimestepLoss:
    """Per timestep: the mean squared error of the predicted noise, with the adapter at full strength and without it, same noise."""

    def __init__(self, backend, timesteps: Sequence[int] = DEFAULT_T, draws: int = 2, seed: int = 0):
        self.backend, self.timesteps, self.draws, self.seed = backend, tuple(timesteps), draws, seed
        self.objective = SDXLObjective()

    def latents(self, images: Sequence[Tensor]) -> list[Tensor]:
        self.backend.stage("decode")  # the autoencoder's stage
        return [self.backend.encode_image(image[None]).cpu() for image in images]

    def run(self, latents: Sequence[Tensor], prompts: Sequence[str] | None = None) -> dict[int, tuple[float, float]]:
        """``{t: (loss without the adapter, loss with it)}`` averaged over the images and draws."""
        prompts = list(prompts) if prompts is not None else [""] * len(latents)
        size = (latents[0].shape[-2] * 8, latents[0].shape[-1] * 8)
        self.backend.stage("encode")
        conditioning = {p: self.backend.condition(p, size) for p in set(prompts)}
        self.backend.stage("sample")
        sums = {t: [0.0, 0.0] for t in self.timesteps}
        count = len(latents) * self.draws
        with torch.no_grad():
            for n, (latent, prompt) in enumerate(zip(latents, prompts, strict=True)):
                for draw in range(self.draws):
                    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(self.seed + 1000 * n + draw))
                    for t in self.timesteps:
                        x_in, target, _ = self.objective.noisy(latent, noise, torch.tensor([t]))
                        for slot, strength in enumerate((0.0, 1.0)):
                            pred = self.backend.predict(x_in, torch.tensor([float(t)]), conditioning[prompt], strength).float().cpu()
                            sums[t][slot] += float((pred - target).pow(2).mean())
        return {t: (a / count, b / count) for t, (a, b) in sums.items()}
