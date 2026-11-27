"""A look at the picture while it forms: what a sampler hands its estimate of the clean latents to.

Sampling Preview builds one and wires it into a Sampler. Every few steps the first image's estimate is shrunk (a decode at full size
needs more memory than a card holding the denoiser may have to spare) and decoded, and the result goes to ``show``. When the decode
does not fit, or fails in any other way, previews stop for the rest of the run and ``note`` says so: a preview must never end the run it watches.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from ..runtime.memory import is_out_of_memory


class SamplingPreview:
    """``decode(latents) -> uint8 (1, 3, H, W)``; ``show(pixels (3, H, W), step, total)``; ``close()`` frees what ``decode`` loaded."""

    def __init__(self, *, every: int, size: int, spatial_factor: int, decode: Callable[[Tensor], Tensor],
                 show: Callable[[Tensor, int, int], None], note: Callable[[str], None], close: Callable[[], None] = lambda: None):
        if every < 1 or size < spatial_factor:
            raise ValueError("a preview needs every >= 1 and a size of at least one latent cell")
        self.every, self.size, self.factor = every, size, spatial_factor
        self.decode, self.show, self.note, self._close = decode, show, note, close
        self.stopped = False

    def __call__(self, step: int, total: int, estimate: Tensor) -> None:
        """``step`` steps are done of ``total``; the last one is always shown."""
        if self.stopped or (step % self.every and step != total):
            return
        try:
            pixels = self.decode(self.shrunk(estimate[:1]))
        except Exception as error:
            self.stopped = True
            self.note("Sampling Preview: the card has no room to decode a preview beside the model; previews stop for this run" if is_out_of_memory(error)
                      else f"Sampling Preview stopped for this run: {type(error).__name__}: {error}")
            self.finish()
            return
        self.show(pixels[0], step, total)

    def shrunk(self, latents: Tensor) -> Tensor:
        """The latents at a size whose decode is about ``size`` pixels on the long side (never larger than they are)."""
        longest = max(latents.shape[-2:]) * self.factor
        if longest <= self.size:
            return latents.detach().float()
        scale = self.size / longest
        height, width = (max(1, round(side * scale)) for side in latents.shape[-2:])
        with torch.no_grad():
            return functional.interpolate(latents.detach().float(), size=(height, width), mode="area")

    def finish(self) -> None:
        """The sampler is done with it: drop the decoder."""
        close, self._close = self._close, lambda: None
        close()
