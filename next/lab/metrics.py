"""Numbers for what the eye sees: how much fine detail an image has, and where a rendering differs from another of the same seed.

Detail is read from the radial power spectrum of the luma in octave bands (the top band is the finest detail the pixels can hold). An
image of "large overlapping brush strokes" has a steeper spectrum (a more negative slope) and less power in the top bands than the
same seed rendered without the adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

OCTAVES = 7  # bands of the spectrum, from half of Nyquist to Nyquist downwards


def luma(pixels: Tensor) -> Tensor:
    """(H, W) float in [0, 1] from (3, H, W) uint8 or float."""
    image = pixels.float() / 255.0 if pixels.dtype == torch.uint8 else pixels.float()
    return 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]


class Spectrum:
    """Mean power per frequency bin in octave bands (index 0 = the finest)."""

    def __init__(self, gray: Tensor):
        height, width = gray.shape
        window = torch.outer(torch.hann_window(height, periodic=False), torch.hann_window(width, periodic=False))
        power = torch.fft.fft2((gray - gray.mean()) * window).abs().pow(2) / window.pow(2).sum()
        fy, fx = torch.fft.fftfreq(height)[:, None], torch.fft.fftfreq(width)[None, :]
        radius = (fx ** 2 + fy ** 2).sqrt()
        bands = []
        for k in range(OCTAVES):
            high, low = 0.5 / 2 ** k, 0.25 / 2 ** k
            mask = (radius > low) & (radius <= high)
            bands.append(power[mask].mean() if mask.any() else torch.tensor(float("nan")))
        self.power = torch.stack(bands)
        self.centres = torch.tensor([0.375 / 2 ** k for k in range(OCTAVES)])

    def slope(self) -> float:
        """Slope of log power against log frequency: natural photographs sit near -2; larger magnitudes mean less fine detail."""
        valid = torch.isfinite(self.power) & (self.power > 0)
        x, y = self.centres[valid].log(), self.power[valid].log()
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / (x * x).sum())

    def fine_share(self) -> float:
        """Share of the power (DC excluded) that lies in the two finest octaves."""
        weight = self.power * torch.tensor([3.0 * 4 ** -k for k in range(OCTAVES)])  # a band holds this share of all bins
        return float(weight[:2].sum() / weight.sum())


def laplacian_variance(gray: Tensor) -> float:
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
    return float(F.conv2d(gray[None, None], kernel).var())


@dataclass(frozen=True)
class Detail:
    slope: float
    fine_share: float
    sharpness: float
    bands: tuple[float, ...]

    @staticmethod
    def of(pixels: Tensor) -> Detail:
        gray = luma(pixels)
        spectrum = Spectrum(gray)
        return Detail(spectrum.slope(), spectrum.fine_share(), laplacian_variance(gray), tuple(float(p) for p in spectrum.power))


@dataclass(frozen=True)
class Change:
    """How a rendering differs from the reference of the same seed, split into what survives a blur and what does not."""

    coarse: float  # mean |difference| of the image reduced 16 times
    fine: float  # mean |difference| of what is left after taking the 8 times blurred image away
    fine_to_coarse: float
    top_band_ratio: float  # power of the finest octave, rendering over reference
    sharpness_ratio: float

    @staticmethod
    def between(pixels: Tensor, reference: Tensor) -> Change:
        a, b = luma(pixels), luma(reference)

        def parts(gray: Tensor) -> tuple[Tensor, Tensor]:
            coarse = F.avg_pool2d(gray[None, None], 16)
            blur = F.interpolate(F.avg_pool2d(gray[None, None], 8), scale_factor=8, mode="bilinear", align_corners=False)
            return coarse, gray[None, None] - blur

        (ca, fa), (cb, fb) = parts(a), parts(b)
        coarse, fine = float((ca - cb).abs().mean()), float((fa - fb).abs().mean())
        top_a, top_b = float(Spectrum(a).power[0]), float(Spectrum(b).power[0])
        return Change(coarse, fine, fine / max(coarse, 1e-9), top_a / max(top_b, 1e-20), laplacian_variance(a) / max(laplacian_variance(b), 1e-20))


def mean(values: list[float]) -> float:
    values = [v for v in values if not math.isnan(v)]
    return sum(values) / len(values) if values else float("nan")
