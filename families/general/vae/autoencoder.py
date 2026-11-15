"""The KL-regularised image autoencoder (latent diffusion; what SDXL ships).

Key names match the published checkpoints (``encoder.*``, ``decoder.*``, ``quant_conv.*``,
``post_quant_conv.*``) so state dicts load without remapping. Two details are easy to get wrong:

* ``Downsample`` pads asymmetrically ``(0, 1, 0, 1)`` before a stride-2 convolution; symmetric
  padding halves the size correctly but shifts the image by half a pixel per level.
* Attention treats channels as the feature axis and ``H*W`` as the sequence (the reverse of the
  UNet's attention), and the middle block always has one whatever ``attn_resolutions`` says.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

SDXL_DDCONFIG = {
    "double_z": True, "z_channels": 4, "resolution": 256, "in_channels": 3, "out_ch": 3, "ch": 128,
    "ch_mult": (1, 2, 4, 4), "num_res_blocks": 2, "attn_resolutions": (), "dropout": 0.0,
}


def group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool, scale_factor: float = 2.0) -> None:
        super().__init__()
        self.with_conv, self.scale_factor = with_conv, scale_factor
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return self.conv(x) if self.with_conv else x


class Downsample(nn.Module):
    def __init__(self, channels: int, with_conv: bool, stride: int = 2) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=stride, padding=0)

    def forward(self, x):
        if not self.with_conv:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int | None = None, conv_shortcut: bool = False,
                 dropout: float = 0.0, temb_channels: int = 0) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels, self.out_channels, self.use_conv_shortcut = in_channels, out_channels, conv_shortcut
        self.swish = nn.SiLU(inplace=True)
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        if temb_channels:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)

    def forward(self, x, temb=None):
        h = self.conv1(self.swish(self.norm1(x)))
        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]
        h = self.conv2(self.dropout(self.swish(self.norm2(h))))
        if self.in_channels == self.out_channels:
            return x + h
        shortcut = self.conv_shortcut if self.use_conv_shortcut else self.nin_shortcut
        return shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = group_norm(channels)
        self.q, self.k, self.v, self.proj_out = (nn.Conv2d(channels, channels, 1) for _ in range(4))

    def forward(self, x):
        h = self.norm(x)
        shape = x.shape
        q, k, v = (t(h).view(shape[0], 1, shape[1], -1).transpose(2, 3).contiguous() for t in (self.q, self.k, self.v))
        out = F.scaled_dot_product_attention(q, k, v)
        return x + self.proj_out(out.transpose(2, 3).reshape(shape))


def _mid(channels: int, dropout: float) -> nn.Module:
    mid = nn.Module()
    mid.block_1 = ResnetBlock(in_channels=channels, dropout=dropout)
    mid.attn_1 = AttnBlock(channels)
    mid.block_2 = ResnetBlock(in_channels=channels, dropout=dropout)
    return mid


class Encoder(nn.Module):
    def __init__(self, *, ch: int, out_ch: int, ch_mult=(1, 2, 4, 8), num_res_blocks: int, attn_resolutions,
                 dropout: float = 0.0, resamp_with_conv: bool = True, in_channels: int, resolution: int,
                 z_channels: int, double_z: bool = True, **_ignored) -> None:
        super().__init__()
        self.num_resolutions, self.num_res_blocks = len(ch_mult), num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, 3, stride=1, padding=1)
        res, widths = resolution, (1, *ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for level in range(self.num_resolutions):
            level_module, blocks, attns = nn.Module(), nn.ModuleList(), nn.ModuleList()
            block_in, block_out = ch * widths[level], ch * ch_mult[level]
            for _ in range(num_res_blocks):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if res in attn_resolutions:
                    attns.append(AttnBlock(block_in))
            level_module.block, level_module.attn = blocks, attns
            if level != self.num_resolutions - 1:
                level_module.downsample = Downsample(block_in, resamp_with_conv)
                res //= 2
            self.down.append(level_module)
        self.mid = _mid(block_in, dropout)
        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels if double_z else z_channels, 3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in range(self.num_resolutions):
            for i in range(self.num_res_blocks):
                h = self.down[level].block[i](h)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[i](h)
            if level != self.num_resolutions - 1:
                h = self.down[level].downsample(h)
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(h)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    """One more residual block per level than the encoder: the widest level is entered, not passed through."""

    def __init__(self, *, ch: int, out_ch: int, ch_mult=(1, 2, 4, 8), num_res_blocks: int, attn_resolutions,
                 dropout: float = 0.0, resamp_with_conv: bool = True, in_channels: int, resolution: int,
                 z_channels: int, tanh_out: bool = False, **_ignored) -> None:
        super().__init__()
        self.num_resolutions, self.num_res_blocks, self.tanh_out = len(ch_mult), num_res_blocks, tanh_out
        block_in = ch * ch_mult[-1]
        res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)
        self.mid = _mid(block_in, dropout)
        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            level_module, blocks, attns = nn.Module(), nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks + 1):
                blocks.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if res in attn_resolutions:
                    attns.append(AttnBlock(block_in))
            level_module.block, level_module.attn = blocks, attns
            if level != 0:
                level_module.upsample = Upsample(block_in, resamp_with_conv)
                res *= 2
            self.up.insert(0, level_module)
        self.norm_out = group_norm(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

    def forward(self, z):
        h = self.mid.block_2(self.mid.attn_1(self.mid.block_1(self.conv_in(z))))
        for level in reversed(range(self.num_resolutions)):
            for i in range(self.num_res_blocks + 1):
                h = self.up[level].block[i](h)
                if len(self.up[level].attn) > 0:
                    h = self.up[level].attn[i](h)
            if level != 0:
                h = self.up[level].upsample(h)
        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h) if self.tanh_out else h


class DiagonalGaussian:
    """Mean and log-variance halves of the encoder output. Log-variance is clamped so a wild output
    cannot overflow ``exp`` into a NaN gradient."""

    def __init__(self, parameters: torch.Tensor) -> None:
        self.parameters = parameters
        self.mean, logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, device=self.mean.device, dtype=self.mean.dtype, generator=generator)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim: int, ddconfig: dict) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**{k: v for k, v in ddconfig.items() if k != "double_z"})
        wide = 2 if ddconfig.get("double_z", False) else 1
        self.quant_conv = nn.Conv2d(wide * ddconfig["z_channels"], wide * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def posterior(self, x: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.quant_conv(self.encoder(x)))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image in [-1, 1] to the posterior mode (unscaled latent)."""
        return self.posterior(x).mode()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.decode(z)
