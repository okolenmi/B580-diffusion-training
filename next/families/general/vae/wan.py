"""The Wan 2.x / Qwen-Image autoencoder (16 latent channels, factor 8) for single images.

The published network is a causal 3-D convolution net for video. One frame sees only the *last* temporal slice of every
3-D kernel (the earlier slices meet the zero padding), and the temporal resampling layers do nothing for a first frame, so an
image runs through plain 2-D layers. ``to_image_state`` turns the file's tensors into that form; the rest of the layout and key
names are the file's.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class _Norm(nn.Module):
    """RMS over channels, then a per-channel gain."""

    def __init__(self, dim: int):
        super().__init__()
        self.scale = math.sqrt(dim)
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=1) * self.scale * self.gamma.to(x)


class _Residual(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.residual = nn.Sequential(_Norm(cin), nn.SiLU(), nn.Conv2d(cin, cout, 3, padding=1),
                                      _Norm(cout), nn.SiLU(), nn.Dropout(0.0), nn.Conv2d(cout, cout, 3, padding=1))
        self.shortcut = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.residual(x) + self.shortcut(x)


class _Attention(nn.Module):
    """One head over all positions of the image."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = _Norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        q, k, v = self.to_qkv(self.norm(x)).reshape(b, 3, 1, c, h * w).transpose(-1, -2).unbind(1)
        out = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj(out)


class _Down(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=2))

    def forward(self, x: Tensor) -> Tensor:
        return self.resample(x)


class _Up(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.resample = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="nearest-exact"), nn.Conv2d(dim, dim // 2, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.resample(x)


class _Encoder(nn.Module):
    def __init__(self, dim: int, out_channels: int, mult: tuple[int, ...], blocks: int):
        super().__init__()
        dims = [dim * u for u in (1, *mult)]
        self.conv1 = nn.Conv2d(3, dims[0], 3, padding=1)
        layers: list[nn.Module] = []
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(blocks):
                layers.append(_Residual(cin, cout))
                cin = cout
            if i != len(mult) - 1:
                layers.append(_Down(cout))
        self.downsamples = nn.Sequential(*layers)
        self.middle = nn.Sequential(_Residual(cout, cout), _Attention(cout), _Residual(cout, cout))
        self.head = nn.Sequential(_Norm(cout), nn.SiLU(), nn.Conv2d(cout, out_channels, 3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.middle(self.downsamples(self.conv1(x))))


class _Decoder(nn.Module):
    def __init__(self, dim: int, z: int, mult: tuple[int, ...], blocks: int):
        super().__init__()
        dims = [dim * u for u in (mult[-1], *mult[::-1])]
        self.conv1 = nn.Conv2d(z, dims[0], 3, padding=1)
        self.middle = nn.Sequential(_Residual(dims[0], dims[0]), _Attention(dims[0]), _Residual(dims[0], dims[0]))
        layers: list[nn.Module] = []
        for i, (cin, cout) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0:
                cin //= 2  # the previous stage's upsampling halved the channels
            for _ in range(blocks + 1):
                layers.append(_Residual(cin, cout))
                cin = cout
            if i != len(mult) - 1:
                layers.append(_Up(cout))
        self.upsamples = nn.Sequential(*layers)
        self.head = nn.Sequential(_Norm(cout), nn.SiLU(), nn.Conv2d(cout, 3, 3, padding=1))

    def forward(self, z: Tensor) -> Tensor:
        return self.head(self.upsamples(self.middle(self.conv1(z))))


class WanImageVAE(nn.Module):
    def __init__(self, dim: int = 96, z_dim: int = 16, mult: tuple[int, ...] = (1, 2, 4, 4), blocks: int = 2):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = _Encoder(dim, z_dim * 2, mult, blocks)
        self.conv1 = nn.Conv2d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = nn.Conv2d(z_dim, z_dim, 1)
        self.decoder = _Decoder(dim, z_dim, mult, blocks)

    def encode(self, x: Tensor) -> Tensor:
        """Images (B, 3, H, W) in [-1, 1] -> the mean of the latent distribution (B, z, H/8, W/8)."""
        return self.conv1(self.encoder(x)).chunk(2, dim=1)[0]

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(self.conv2(z))


def to_image_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """The file's video tensors in the layout of ``WanImageVAE``: 3-D kernels keep their last temporal slice, gains lose
    their time axis, and the temporal resampling convolutions (``time_conv``) are dropped."""
    out: dict[str, Tensor] = {}
    for key, value in state.items():
        if ".time_conv." in key:
            continue
        if value.ndim == 5:
            value = value[:, :, -1].contiguous()
        elif key.endswith(".gamma") and value.ndim == 4:
            value = value[..., 0]
        out[key] = value
    return out
