"""Decoding a large latent in overlapping tiles blended with linear ramps: the way out when the whole image does not fit the device."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


class TiledDecoder:
    def __init__(self, decode: Callable[[Tensor], Tensor], scale: int = 8, tile: int = 64, overlap: int = 16):
        if overlap * 2 > tile:
            raise ValueError("the overlap must be at most half a tile")
        self.decode, self.scale, self.tile, self.overlap = decode, scale, tile, overlap

    def _starts(self, size: int) -> list[int]:
        if size <= self.tile:
            return [0]
        starts = list(range(0, size - self.tile, self.tile - self.overlap))
        return starts + [size - self.tile]

    def _ramp(self, length: int, first: bool, last: bool) -> Tensor:
        weight = torch.ones(length)
        fade = torch.linspace(0, 1, self.overlap * self.scale + 2)[1:-1]
        if not first:
            weight[: len(fade)] = fade
        if not last:
            weight[-len(fade):] = torch.minimum(weight[-len(fade):], fade.flip(0))
        return weight

    def __call__(self, latents: Tensor) -> Tensor:
        """(1, C, H, W) latents -> (1, 3, H*scale, W*scale) uint8."""
        height, width = latents.shape[-2:]
        rows, cols = self._starts(height), self._starts(width)
        out = torch.zeros(1, 3, height * self.scale, width * self.scale)
        total = torch.zeros(1, 1, height * self.scale, width * self.scale)
        for r, top in enumerate(rows):
            for c, left in enumerate(cols):
                piece = self.decode(latents[:, :, top:top + self.tile, left:left + self.tile]).float()
                h, w = piece.shape[-2:]
                mask = (self._ramp(h, r == 0, r == len(rows) - 1)[:, None] * self._ramp(w, c == 0, c == len(cols) - 1)[None, :])[None, None]
                y, x = top * self.scale, left * self.scale
                out[:, :, y:y + h, x:x + w] += piece * mask
                total[:, :, y:y + h, x:x + w] += mask
        return (out / total).round().clamp(0, 255).to(torch.uint8)
