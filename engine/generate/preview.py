"""A look at the picture while it forms: what a sampler hands its estimate of the clean latents to.

``LivePreview`` is the object a Sampler uses (its ``preview`` addon slot). Every few steps it turns the first image's estimate into a small picture
with a ``PreviewDecoder`` and hands that to ``show``. Two decoders: ``LatentRGB`` reads the picture off the latents with one linear map (no weights, no
memory, a blurry result) and ``VAEPreview`` runs the real decoder on shrunk latents (accurate, needs the VAE and some memory beside the denoiser).
A preview must never end the run it watches: when one fails, previews stop for the rest of the run and ``note`` says why.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..runtime.memory import is_out_of_memory


class PreviewUnavailable(Exception):
    """A preview that cannot be made here (no projection for this family, no VAE); the message is what the person is told."""


class PreviewDecoder(ABC):
    @abstractmethod
    def decode(self, latents: Tensor) -> Tensor:
        """``latents`` (1, C, h, w) in the sampler's space -> uint8 ``(3, H, W)``."""

    def close(self) -> None:
        """Give back what ``decode`` loaded."""


class LatentRGB(PreviewDecoder):
    """``rgb = (latents @ factors + bias + 1) / 2``: a colour guess per latent cell, so the picture is as small as the latents (an eighth of the final one)."""

    def __init__(self, factors: tuple[tuple[float, ...], ...], bias: tuple[float, ...] = ()):
        self.factors = torch.tensor(factors, dtype=torch.float32)
        self.bias = torch.tensor(bias or (0.0, 0.0, 0.0), dtype=torch.float32)

    def decode(self, latents: Tensor) -> Tensor:
        cells = latents[0].detach().float().permute(1, 2, 0).cpu()
        if cells.shape[-1] != self.factors.shape[0]:
            raise ValueError(f"the projection is for {self.factors.shape[0]} channels, the latents have {cells.shape[-1]}")
        rgb = cells @ self.factors + self.bias
        return ((rgb + 1.0) / 2.0).clamp(0, 1).mul(255).round().to(torch.uint8).permute(2, 0, 1).contiguous()


class VAEPreview(PreviewDecoder):
    """The family's decoder on latents shrunk until their decode is about ``size`` pixels on the long side (a decode at full size
    needs more memory than a card holding the denoiser may have to spare). ``load()`` runs at the first preview, ``release()`` ends it."""

    def __init__(self, *, size: int, spatial_factor: int, load: Callable[[], Any], release: Callable[[Any], None] = lambda codec: None):
        if size < spatial_factor:
            raise ValueError("a preview needs a size of at least one latent cell")
        self.size, self.factor, self._load, self._release = size, spatial_factor, load, release
        self._codec: Any = None

    def decode(self, latents: Tensor) -> Tensor:
        if self._codec is None:
            self._codec = self._load()
        with torch.no_grad():
            return self._codec.decode(self.shrunk(latents))[0]

    def shrunk(self, latents: Tensor) -> Tensor:
        """The latents at a size whose decode is about ``size`` pixels on the long side (never larger than they are)."""
        longest = max(latents.shape[-2:]) * self.factor
        if longest <= self.size:
            return latents.detach().float()
        scale = self.size / longest
        height, width = (max(1, round(side * scale)) for side in latents.shape[-2:])
        with torch.no_grad():
            return functional.interpolate(latents.detach().float(), size=(height, width), mode="area")

    def close(self) -> None:
        codec, self._codec = self._codec, None
        if codec is not None:
            self._release(codec)


class LivePreview:
    """What the sampler calls: ``begin`` once with its family, ``__call__(step, total, estimate)`` after each step, ``finish`` at the end.

    ``choose(family)`` returns the decoder to use, None when nobody would see a preview (no node downstream shows one: then nothing is computed),
    or raises ``PreviewUnavailable``. ``show(pixels (3, H, W), step, total)`` receives each picture.
    """

    def __init__(self, *, every: int, choose: Callable[[Any], PreviewDecoder | None], show: Callable[[Tensor, int, int], None], note: Callable[[str], None]):
        if every < 1:
            raise ValueError("a preview needs every >= 1")
        self.every, self.choose, self.show, self.note = every, choose, show, note
        self.decoder: PreviewDecoder | None = None

    @property
    def active(self) -> bool:
        return self.decoder is not None

    def begin(self, family: Any) -> None:
        try:
            self.decoder = self.choose(family)
        except PreviewUnavailable as reason:
            self.note(f"Live preview: {reason}")
        except Exception as error:  # noqa: BLE001 -- a preview must not end the run
            self.note(f"Live preview stopped for this run: {type(error).__name__}: {error}")

    def __call__(self, step: int, total: int, estimate: Tensor) -> None:
        """``step`` steps are done of ``total``; the last one is always shown."""
        if self.decoder is None or (step % self.every and step != total):
            return
        try:
            pixels = self.decoder.decode(estimate[:1])
            self.show(pixels, step, total)
        except Exception as error:  # noqa: BLE001
            self.note("Live preview: the card has no room to decode a preview beside the model; previews stop for this run" if is_out_of_memory(error)
                      else f"Live preview stopped for this run: {type(error).__name__}: {error}")
            self.finish()

    def finish(self) -> None:
        """The sampler is done with it: stop and drop the decoder's memory."""
        decoder, self.decoder = self.decoder, None
        if decoder is not None:
            decoder.close()
