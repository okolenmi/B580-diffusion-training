"""Encoding images to scaled latents and decoding them back, around any autoencoder that has
``encode`` / ``decode`` on [-1, 1] images."""

from __future__ import annotations

import gc
from dataclasses import dataclass
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
    mean: tuple[float, ...] = ()   # per-channel offset and spread, applied after shift and scale when given
    std: tuple[float, ...] = ()
    grid: int = 1                  # the network needs latent sides divisible by this (a transformer's patch size)
    rgb: tuple[tuple[float, ...], ...] = ()  # one (r, g, b) row per channel: a linear guess of the picture from the latents, for live previews
    rgb_bias: tuple[float, ...] = ()

    @property
    def pixel_multiple(self) -> int:
        """Images must be cut to a multiple of this many pixels."""
        return self.spatial_factor * self.grid

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

    @staticmethod
    def _channels(values: tuple[float, ...], like: Tensor) -> Tensor:
        return torch.tensor(values, device=like.device, dtype=like.dtype).view(1, -1, 1, 1)

    @torch.no_grad()
    def encode(self, images: Tensor) -> Tensor:
        """Images (B,3,H,W) as float in [0,1] or uint8 -> scaled latents on the codec's device."""
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        raw = self.model.encode(images.to(self.device, self.dtype) * 2.0 - 1.0)
        latents = (raw - self.spec.shift) * self.spec.scale
        if self.spec.mean:
            latents = (latents - self._channels(self.spec.mean, latents)) / self._channels(self.spec.std, latents)
        return latents

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Scaled latents -> uint8 images (B,3,H,W) on the CPU."""
        latents = latents.to(self.device, self.dtype)
        if self.spec.mean:
            latents = latents * self._channels(self.spec.std, latents) + self._channels(self.spec.mean, latents)
        raw = latents / self.spec.scale + self.spec.shift
        pixels = self.model.decode(raw).float().clamp(-1, 1)
        return ((pixels + 1.0) / 2.0 * 255.0).round().to(torch.uint8).cpu()

    def release(self) -> None:
        """Drop the weights. The caller empties the device cache."""
        self.model.to("cpu")
        self.model = None
        gc.collect()
