"""Encoding images to scaled latents and decoding them back, around any autoencoder that has
``encode`` / ``decode`` on [-1, 1] images."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .autoencoder import AutoencoderKL


@dataclass(frozen=True)
class LatentSpec:
    """The latent space a family trains in."""

    channels: int
    spatial_factor: int
    scale: float
    shift: float = 0.0

    def noise_shape(self, height: int, width: int, batch: int = 1) -> tuple[int, ...]:
        return (batch, self.channels, height // self.spatial_factor, width // self.spatial_factor)


class VAECodec:
    """Latents are ``(raw - shift) * scale`` on the way in and the inverse on the way out."""

    def __init__(self, model: nn.Module, spec: LatentSpec, device="cpu", dtype: torch.dtype = torch.float32):
        self.model, self.spec, self.device, self.dtype = model, spec, torch.device(device), dtype
        self.model.eval().requires_grad_(False)
        self.model.to(device=self.device, dtype=dtype)

    @classmethod
    def kl_from_state_dict(cls, state: Mapping[str, Tensor], ddconfig: dict, embed_dim: int, spec: LatentSpec,
                           device="cpu", dtype: torch.dtype = torch.float32) -> VAECodec:
        """Build the KL autoencoder and load ``state`` strictly: a VAE that does not fit is an error."""
        model = AutoencoderKL(embed_dim, ddconfig)
        result = model.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"VAE weights do not fit: {len(result.missing_keys)} missing "
                             f"(e.g. {result.missing_keys[:2]}), {len(result.unexpected_keys)} unexpected "
                             f"(e.g. {result.unexpected_keys[:2]})")
        return cls(model, spec, device, dtype)

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        """Images (B,3,H,W) as float in [0,1] or uint8 -> scaled latents on the codec's device."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        raw = self.model.encode(images.to(self.device, self.dtype) * 2.0 - 1.0)
        return (raw - self.spec.shift) * self.spec.scale

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Scaled latents -> uint8 images (B,3,H,W) on the CPU."""
        raw = latents.to(self.device, self.dtype) / self.spec.scale + self.spec.shift
        pixels = self.model.decode(raw).float().clamp(-1, 1)
        return ((pixels + 1.0) / 2.0 * 255.0).round().to(torch.uint8).cpu()

    def save_images(self, latents: Tensor | list[Tensor], directory: str | Path, prefix: str = "sample",
                    start: int = 0, batch_size: int = 1) -> int:
        from PIL import Image

        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        if isinstance(latents, list):
            latents = torch.cat(latents)
        saved = 0
        for first in range(0, latents.shape[0], batch_size):
            for image in self.decode(latents[first:first + batch_size]):
                Image.fromarray(image.permute(1, 2, 0).numpy()).save(out / f"{prefix}_{start + saved:03d}.png")
                saved += 1
        return saved

    def release(self) -> None:
        """Drop the weights. The caller empties the device cache."""
        self.model.to("cpu")
        self.model = None
        gc.collect()
