"""Things made to be looked at: a labelled grid of images, and the padding sheet that shows what zero and mirror padding put in the strip around a sample."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor

from ..data.padding import mirror_fill


def to_pil(image: Image.Image | Tensor) -> Image.Image:
    """A PIL image, or a (C, H, W) tensor in 0..1 (1 or 3 channels) turned into one."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    data = image.detach().float().clamp(0, 1).cpu()
    if data.shape[0] == 1:
        data = data.expand(3, -1, -1)
    return Image.fromarray((data.permute(1, 2, 0) * 255).round().byte().numpy())


class ImageSheet:
    """Images in a grid of equal cells (each scaled to fit, aspect kept), a caption under each; saved as one PNG."""

    def __init__(self, cell: tuple[int, int] = (256, 256), columns: int = 3, caption_height: int = 18, gap: int = 6, title: str = ""):
        self.cell, self.columns, self.caption_height, self.gap, self.title = cell, columns, caption_height, gap, title
        self.items: list[tuple[Image.Image, str]] = []

    def add(self, image: Image.Image | Tensor, caption: str = "") -> "ImageSheet":
        self.items.append((to_pil(image), caption))
        return self

    def render(self) -> Image.Image:
        columns = max(1, min(self.columns, len(self.items)))
        rows = -(-len(self.items) // columns)
        width, height = self.cell
        top = 22 if self.title else 0
        row_height = height + self.caption_height + self.gap
        sheet = Image.new("RGB", (columns * (width + self.gap) + self.gap, top + rows * row_height + self.gap), (32, 32, 32))
        draw = ImageDraw.Draw(sheet)
        if self.title:
            draw.text((self.gap, 4), self.title, fill=(235, 235, 235))
        for i, (image, caption) in enumerate(self.items):
            x, y = self.gap + (i % columns) * (width + self.gap), top + self.gap + (i // columns) * row_height
            fitted = image.copy()
            fitted.thumbnail(self.cell)
            sheet.paste(fitted, (x + (width - fitted.width) // 2, y + (height - fitted.height) // 2))
            draw.text((x, y + height + 2), caption[: width // 6], fill=(220, 220, 150))
        return sheet

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render().save(path)
        return path


class PaddingSheet:
    """For each sample: where it sits on the canvas, then the same noisy canvas with the strip left as noise (zero padding: the padded part of the clean
    input is zero, so after noising it is noise alone) and with the strip filled by the reflection of the noisy sample (mirror padding). Pixels stand in
    for latents; the layout is the same at one eighth the size."""

    def __init__(self, canvas: tuple[int, int], sigma: float = 0.6, seed: int = 0, cell: int = 256):
        self.canvas, self.sigma, self.seed, self.cell = canvas, sigma, seed, cell

    def place(self, image: Image.Image, rng: random.Random) -> tuple[Tensor, tuple[int, int, int, int]]:
        height, width = self.canvas
        scale = min(height / image.height, width / image.width, 1.0) * rng.uniform(0.55, 0.9)  # a sample smaller than the canvas, as in a bucket
        fitted = image.convert("RGB").resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        data = torch.from_numpy(np.asarray(fitted).copy()).permute(2, 0, 1).float() / 255.0
        h, w = data.shape[-2:]
        top, left = rng.randint(0, max(height - h, 0)), rng.randint(0, max(width - w, 0))
        canvas = torch.zeros(3, height, width)
        canvas[:, top:top + h, left:left + w] = data
        return canvas, (top, left, h, w)

    def sheet(self, images: list[Image.Image]) -> ImageSheet:
        out = ImageSheet((self.cell, self.cell), 3, title=f"zero against mirror padding at noise level {self.sigma} (pixels stand in for latents)")
        rng = random.Random(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        for i, image in enumerate(images):
            clean, (top, left, h, w) = self.place(image, rng)
            noise = torch.randn(clean.shape, generator=generator)
            noisy = (1 - self.sigma) * clean + self.sigma * noise
            zero = noisy  # the padded part of the clean canvas is zero, so it is noise alone, as in training
            mirror = mirror_fill(noisy, top, left, h, w)
            shown = lambda t: ((t - 0.5) * 0.5 + 0.5).clamp(0, 1)  # noise around zero shown around grey  # noqa: E731
            out.add(clean, f"{i + 1}: clean, on the canvas")
            out.add(shown(zero), f"{i + 1}: zero padding (strip is noise)")
            out.add(shown(mirror), f"{i + 1}: mirror padding")
        return out
