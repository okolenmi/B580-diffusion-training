"""Rendering a grid of images (prompt x seed x strength) with one model, and keeping what produced each image with it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import Tensor

from .sampling import SAMPLERS, GuidedDenoiser, NoiseTable, SigmaSchedule, initial_noise


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


@dataclass(frozen=True)
class Grid:
    """Prompts, seeds and strengths to cross, with the sampling settings they share."""

    prompts: tuple[str, ...]
    seeds: tuple[int, ...] = (1, 2, 3, 4)
    strengths: tuple[float, ...] = (0.5, 1.0)
    negative: str = ""
    steps: int = 25
    cfg: float = 6.0
    sampler: str = "dpmpp_2m"
    scheduler: str = "normal"
    width: int = 1024
    height: int = 1024

    def recipes(self, strengths: Sequence[float] | None = None) -> list[Recipe]:
        shared = {k: getattr(self, k) for k in ("negative", "steps", "cfg", "sampler", "scheduler", "width", "height")}
        return [Recipe(prompt=p, seed=s, strength=x, **shared) for p in self.prompts for s in self.seeds for x in (strengths if strengths is not None else self.strengths)]


class Backend(Protocol):
    """What rendering needs from a model. ``stage`` lets a backend keep only what the current stage uses on the device."""

    def stage(self, name: str) -> None: ...

    def condition(self, prompt: str, size: tuple[int, int]) -> Any: ...

    def predict(self, x_in: Tensor, t: Tensor, conditioning: Any, strength: float) -> Tensor: ...

    def decode(self, latents: Tensor) -> Tensor: ...


@dataclass
class Image:
    recipe: Recipe
    pixels: Tensor  # (3, H, W) uint8
    meta: dict = field(default_factory=dict)


class Renderer:
    """Samples every recipe's latent first and decodes afterwards, so the network and the decoder never share the device."""

    def __init__(self, backend: Backend, table: NoiseTable | None = None, device: str = "cpu"):
        self.backend, self.table, self.device = backend, table or NoiseTable(), device

    def _latent(self, recipe: Recipe, conditionings: dict) -> Tensor:
        size = (recipe.height, recipe.width)
        positive, negative = conditionings[(recipe.prompt, size)], conditionings[(recipe.negative, size)]

        def network(x_in: Tensor, t: Tensor, is_positive: bool) -> Tensor:
            return self.backend.predict(x_in, t, positive if is_positive else negative, recipe.strength).float()

        sigmas = SigmaSchedule(recipe.scheduler, self.table).sigmas(recipe.steps)
        x = initial_noise(recipe.seed, (1, 4, recipe.height // 8, recipe.width // 8), float(sigmas[0]), self.table).to(self.device)
        with torch.no_grad():
            cfg = 1.0 if recipe.prompt == recipe.negative else recipe.cfg  # equal prompts predict equal noise: guidance changes nothing, the second pass is wasted
            return SAMPLERS[recipe.sampler]().run(GuidedDenoiser(network, self.table, cfg, recipe.eps_scale), x, sigmas).cpu()  # sigmas stay on the host: 0-dim host tensors combine with device tensors

    def render(self, recipes: Iterable[Recipe], meta: dict | None = None, on_image=None) -> list[Image]:
        recipes = list(recipes)
        self.backend.stage("encode")
        conditionings = {}
        for recipe in recipes:
            for prompt in (recipe.prompt, recipe.negative):
                key = (prompt, (recipe.height, recipe.width))
                if key not in conditionings:
                    conditionings[key] = self.backend.condition(prompt, key[1])
        self.backend.stage("sample")
        latents = [self._latent(recipe, conditionings) for recipe in recipes]
        self.backend.stage("decode")
        images = []
        for recipe, latent in zip(recipes, latents, strict=True):
            image = Image(recipe, self.backend.decode(latent)[0], {"recipe": asdict(recipe), **(meta or {})})
            images.append(image)
            if on_image is not None:
                on_image(image)
        return images


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
