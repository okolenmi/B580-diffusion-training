"""Turning an image file into the crops a latent cache encodes.

A *size policy* is the part of ingestion that decides pixels: base resolution, how the image is fitted
to it, and the multiple every side must be a multiple of. It is part of a cache entry's key, so
changing it builds new latents instead of mixing sizes silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

CROP_MODES = ("fit", "center_crop", "pad", "stretch")


@dataclass(frozen=True)
class SizePolicy:
    base: int = 1024
    mode: str = "fit"
    multiple: int = 8
    max_aspect: float = 2.0  # fit only: longer images are split into several crops

    def __post_init__(self) -> None:
        if self.mode not in CROP_MODES:
            raise ValueError(f"unknown crop mode {self.mode!r}; choose from {CROP_MODES}")
        if self.base < self.multiple or self.base % self.multiple:
            raise ValueError(f"base resolution {self.base} must be a positive multiple of {self.multiple}")
        if self.max_aspect < 1.0:
            raise ValueError("max_aspect must be at least 1")

    @property
    def key(self) -> str:
        return f"{self.mode}-{self.base}-m{self.multiple}-a{self.max_aspect:g}"


def _floor(n: int, multiple: int) -> int:
    return max(multiple, n - n % multiple)


def fit_boxes(width: int, height: int, policy: SizePolicy) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    """(resized size, crop boxes) for the ``fit`` mode: short side to ``base``, long side cropped.

    A long side beyond ``max_aspect * base`` is cut into equal segments, each a separate sample.
    """
    scale = policy.base / min(width, height)
    new_w, new_h = round(width * scale), round(height * scale)
    cap = round(policy.max_aspect * policy.base)
    long_side = max(new_w, new_h)
    count = max(1, math.ceil(long_side / cap))
    seg = long_side / count
    boxes = []
    for i in range(count):
        start = round(i * seg)
        end = long_side if i == count - 1 else round((i + 1) * seg)
        box = (start, 0, end, new_h) if new_w >= new_h else (0, start, new_w, end)
        bw, bh = box[2] - box[0], box[3] - box[1]
        aw, ah = _floor(bw, policy.multiple), _floor(bh, policy.multiple)
        left, top = box[0] + (bw - aw) // 2, box[1] + (bh - ah) // 2
        boxes.append((left, top, left + aw, top + ah))
    return (new_w, new_h), boxes


def crop_count(width: int, height: int, policy: SizePolicy) -> int:
    return len(fit_boxes(width, height, policy)[1]) if policy.mode == "fit" else 1


def load_crops(path: str | Path, policy: SizePolicy):
    """The crops of one image as uint8 arrays (H, W, 3), in crop order."""
    import numpy as np
    from PIL import Image

    lanczos = Image.Resampling.LANCZOS
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    w, h = image.size
    base = policy.base
    if policy.mode == "fit":
        size, boxes = fit_boxes(w, h, policy)
        resized = image.resize(size, lanczos)
        return [np.asarray(resized.crop(box)) for box in boxes]
    if policy.mode == "center_crop":
        scale = base / min(w, h)
        resized = image.resize((round(w * scale), round(h * scale)), lanczos)
        left, top = (resized.width - base) // 2, (resized.height - base) // 2
        return [np.asarray(resized.crop((left, top, left + base, top + base)))]
    if policy.mode == "pad":
        scale = base / max(w, h)
        resized = image.resize((round(w * scale), round(h * scale)), lanczos)
        canvas = Image.new("RGB", (base, base), (0, 0, 0))
        canvas.paste(resized, ((base - resized.width) // 2, (base - resized.height) // 2))
        return [np.asarray(canvas)]
    return [np.asarray(image.resize((base, base), lanczos))]
