"""What decides one image, the image itself, and the PNG store that keeps both: shared by the lab's grids and the generation nodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from torch import Tensor


@dataclass(frozen=True)
class Recipe:
    """Everything that decides one image (given the model and the adapter file)."""

    prompt: str
    seed: int = 0
    strength: float = 1.0
    negative: str = ""
    steps: int = 25
    cfg: float = 6.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "normal"
    width: int = 1024
    height: int = 1024
    eps_scale: float = 1.0

    @property
    def stem(self) -> str:
        base = f"p{hashlib.md5(self.prompt.encode()).hexdigest()[:6]}_s{self.seed}_x{self.strength:g}"
        return base if self.eps_scale == 1.0 else f"{base}_e{self.eps_scale:g}"


@dataclass
class Image:
    recipe: Recipe
    pixels: Tensor  # (3, H, W) uint8
    meta: dict = field(default_factory=dict)




class ImageStore:
    """Images as PNG files whose text chunk holds the JSON of what made them, and an ``index.json`` listing them all."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []

    def save(self, image: Image) -> Path:
        from PIL import Image as PILImage
        from PIL.PngImagePlugin import PngInfo

        path = self.directory / f"{image.recipe.stem}.png"
        info = PngInfo()
        info.add_text("lab", json.dumps(image.meta, sort_keys=True))
        info.add_text("parameters", f"{image.recipe.prompt}\nNegative prompt: {image.recipe.negative}\nSteps: {image.recipe.steps}, Sampler: {image.recipe.sampler}, "
                                    f"Schedule type: {image.recipe.scheduler}, CFG scale: {image.recipe.cfg}, Seed: {image.recipe.seed}, "
                                    f"Size: {image.recipe.width}x{image.recipe.height}, Lora strength: {image.recipe.strength}, Epsilon scale: {image.recipe.eps_scale:g}")
        PILImage.fromarray(image.pixels.permute(1, 2, 0).numpy()).save(path, pnginfo=info)
        self.entries.append({"file": path.name, **image.meta})
        (self.directory / "index.json").write_text(json.dumps(self.entries, indent=1, sort_keys=True), encoding="utf8")
        return path

    @staticmethod
    def read_meta(path: str | Path) -> dict:
        from PIL import Image as PILImage

        with PILImage.open(path) as handle:
            return json.loads(handle.text["lab"])
