"""The SDXL VAE, owned rather than imported from ComfyUI.

Design doc 12, section 7.3, section B. `Encoder`, `Decoder`, `ResnetBlock`,
`AttnBlock`, `Upsample`, `Downsample`, `DiagonalGaussianDistribution` and
`AutoencoderKL` -- the published KL-regularised autoencoder from
*High-Resolution Image Synthesis with Latent Diffusion*, which is what SDXL
(and Stable Diffusion before it) ships.

ComfyUI splits this across two files: `comfy/ldm/models/autoencoder.py` has
the engine classes and `comfy/ldm/modules/diffusionmodules/model.py` has the
encoder and decoder. Both are here.

**The contract is again the checkpoint.** `vae_decode.py` builds this from a
`ddconfig` and loads a state dict whose keys are `encoder.*`, `decoder.*`,
`quant_conv.*`, `post_quant_conv.*`.

**What is not ported.** Most of `model.py` is video, and none of it applies
to an image VAE: `conv3d`, `CarriedConv3d`, `VideoConv3d`,
`conv_carry_causal_3d`, `torch_cat_if_needed`, `interpolate_up`,
`time_compress`, the 5-D branches in `Upsample`/`Downsample`, and the
causal-carry plumbing threaded through every `forward` as
`conv_carry_in`/`conv_carry_out`. Also dropped: the `AbstractAutoencoder`
base with its EMA bookkeeping (`LitEma`), `get_input`,
`instantiate_optimizer_from_config`, `configure_optimizers`,
`on_train_batch_end`, and the whole `instantiate_from_config`
string-target dispatch the engine classes are built on -- ours takes the
config as keyword arguments, which is what the only caller does.

`max_batch_size`, `batch_norm_latent`, `decoder_ddconfig` and `conv_shortcut`
are kept as parameters where they change behaviour, because they are part of
the published spec rather than ComfyUI plumbing. SDXL sets none of them.

**Two details that are easy to get wrong and are reproduced deliberately:**

* `Downsample` pads *asymmetrically* -- `(0, 1, 0, 1)` -- before a stride-2
  convolution. `torch.nn.Conv2d` cannot express it, so a symmetric-padding
  version halves the size correctly but shifts the image by half a pixel per
  level. Over four levels that is a visible offset.
* `AttnBlock`'s attention treats **channels as the feature dimension** and
  `H*W` as the sequence: `[B, C, H, W]` becomes `[B, 1, H*W, C]`. That is
  the opposite convention from the UNet's attention in `attention.py`, which
  splits heads out of the channel axis. Getting them the same way round is a
  silent, plausible-looking error, because both produce the right shape.

`mid.attn_1` is built unconditionally in both the encoder and the decoder,
whatever `attn_resolutions` says -- and `attn_resolutions` is empty for SDXL.
So `AttnBlock` is used even by a VAE configured with no attention, which is
worth stating because it makes `AttnBlock` non-optional here.

Provenance: follows ComfyUI's `comfy/ldm/models/autoencoder.py` and
`comfy/ldm/modules/diffusionmodules/model.py` (Apache-2.0), which follow the
published architecture.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "AttnBlock",
    "AutoencoderKL",
    "Decoder",
    "DiagonalGaussianDistribution",
    "Encoder",
    "ResnetBlock",
    "group_norm_32",
]


def group_norm_32(in_channels: int) -> nn.GroupNorm:
    """32-group GroupNorm, eps 1e-6, affine -- the VAE's normalisation."""
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6,
                        affine=True)


def _silu(x: torch.Tensor) -> torch.Tensor:
    return F.silu(x)


class Upsample(nn.Module):
    """Nearest-neighbour upsampling by `scale_factor`, then a convolution."""

    def __init__(self, in_channels: int, with_conv: bool,
                 scale_factor: float = 2.0) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.scale_factor = scale_factor
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=1,
                                  padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Halve the spatial size with a strided convolution or average pooling."""

    def __init__(self, in_channels: int, with_conv: bool,
                 stride: int = 2) -> None:
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            # No asymmetric padding in torch conv, so it is done by hand --
            # see the module docstring.
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=stride,
                                  padding=0)

    def forward(self, x):
        if not self.with_conv:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    """The VAE's residual block: GroupNorm -> SiLU -> conv, twice, plus skip.

    No timestep conditioning -- `temb_channels` is 0 for the image VAE and the
    parameter is kept so the block's shape is the published one.
    """

    def __init__(self, *, in_channels: int, out_channels: int | None = None,
                 conv_shortcut: bool = False, dropout: float = 0.0,
                 temb_channels: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.swish = nn.SiLU(inplace=True)
        self.norm1 = group_norm_32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=1,
                               padding=1)
        if temb_channels:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = group_norm_32(out_channels)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3,
                                               stride=1, padding=1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1,
                                              stride=1, padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.swish(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels == self.out_channels:
            return x + h
        # Two different shortcuts: a 1x1 `nin_shortcut` normally, or a 3x3
        # `conv_shortcut` when conv_shortcut=True. They are not
        # interchangeable and neither exists when the widths match.
        shortcut = self.conv_shortcut if self.use_conv_shortcut \
            else self.nin_shortcut
        return shortcut(x) + h


def _vae_attention(q: torch.Tensor, k: torch.Tensor,
                   v: torch.Tensor) -> torch.Tensor:
    """Self-attention over the spatial axes of a `[B, C, H, W]` tensor.

    Channels are the feature dimension and `H*W` the sequence -- the reverse
    of the UNet's attention. See the module docstring; the two are easy to
    confuse and a mix-up still produces the right shape.
    """
    original = q.shape
    batch, channels = original[0], original[1]
    # [B, C, H, W] -> [B, 1, H*W, C]
    q, k, v = (t.view(batch, 1, channels, -1).transpose(2, 3).contiguous()
               for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                         dropout_p=0.0, is_causal=False)
    return out.transpose(2, 3).reshape(original)


class AttnBlock(nn.Module):
    """Single-head spatial self-attention with a 1x1 convolution on each side.

    Built unconditionally in both halves of the VAE, even when
    `attn_resolutions` is empty, so SDXL has four of them (one per encoder and
    decoder middle block).
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = group_norm_32(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        h = self.norm(x)
        h = _vae_attention(self.q(h), self.k(h), self.v(h))
        return x + self.proj_out(h)


class Encoder(nn.Module):
    """Image -> 2 * z_channels, for the KL split.

    `double_z=True` makes the last convolution emit mean and log-variance
    concatenated; `z_channels` (4 for SDXL) is the latent width each takes.
    """

    def __init__(self, *, ch: int, out_ch: int, ch_mult=(1, 2, 4, 8),
                 num_res_blocks: int, attn_resolutions, dropout: float = 0.0,
                 resamp_with_conv: bool = True, in_channels: int,
                 resolution: int, z_channels: int, double_z: bool = True,
                 **ignore) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        self.conv_in = nn.Conv2d(in_channels, self.ch, 3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0, dropout=dropout)
        # Unconditional, unlike the per-level `attn` lists above.
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0, dropout=dropout)

        self.norm_out = group_norm_32(block_in)
        self.conv_out = nn.Conv2d(
            block_in, 2 * z_channels if double_z else z_channels,
            3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[level].block[i_block](h)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[i_block](h)
            if level != self.num_resolutions - 1:
                h = self.down[level].downsample(h)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        h = self.norm_out(h)
        return self.conv_out(_silu(h))


class Decoder(nn.Module):
    """z_channels -> image.

    One more resnet block per level than the encoder (`num_res_blocks + 1`),
    because the widest level is entered rather than passed through.
    """

    def __init__(self, *, ch: int, out_ch: int, ch_mult=(1, 2, 4, 8),
                 num_res_blocks: int, attn_resolutions, dropout: float = 0.0,
                 resamp_with_conv: bool = True, in_channels: int,
                 resolution: int, z_channels: int, tanh_out: bool = False,
                 **ignore) -> None:
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.tanh_out = tanh_out

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, 3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=0, dropout=dropout)

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=0, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            # Prepended, so the levels come out in ascending order.
            self.up.insert(0, up)

        self.norm_out = group_norm_32(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, 3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[level].block[i_block](h)
                if len(self.up[level].attn) > 0:
                    h = self.up[level].attn[i_block](h)
            if level != 0:
                h = self.up[level].upsample(h)

        h = self.norm_out(h)
        h = self.conv_out(_silu(h))
        return torch.tanh(h) if self.tanh_out else h


class DiagonalGaussianDistribution:
    """The KL posterior: a 2 * z_channels tensor split into mean and logvar.

    `logvar` is clamped to [-30, 20] before `exp`, which is what keeps a
    wild encoder output from producing an infinity and then a NaN gradient.
    """

    def __init__(self, parameters: torch.Tensor,
                 deterministic: bool = False) -> None:
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, device=self.parameters.device,
                            dtype=self.parameters.dtype)
        return self.mean + self.std * noise

    def mode(self) -> torch.Tensor:
        return self.mean

    def kl(self) -> torch.Tensor:
        if self.deterministic:
            return torch.tensor([0.0])
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
            dim=[1, 2, 3])


class AutoencoderKL(nn.Module):
    """The KL-regularised autoencoder SDXL ships.

    :param embed_dim: latent width (4 for SDXL).
    :param ddconfig: the published config -- `ch`, `ch_mult`, `num_res_blocks`,
        `in_channels`, `out_ch`, `resolution`, `z_channels`, `double_z`,
        `attn_resolutions`, `dropout`, `resamp_with_conv`.
    """

    def __init__(self, embed_dim: int, ddconfig: dict) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.encoder = Encoder(**ddconfig)
        decoder_config = dict(ddconfig)
        decoder_config.pop("double_z", None)
        self.decoder = Decoder(**decoder_config)

        double_z = ddconfig.get("double_z", False)
        z_channels = ddconfig["z_channels"]
        # A 1x1 convolution either side of the latent: quant_conv folds the
        # KL pair back down to embed_dim, post_quant_conv widens it again for
        # the decoder. For SDXL both are 8->4 and 4->4.
        self.quant_conv = nn.Conv2d(
            (1 + double_z) * z_channels, (1 + double_z) * embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(embed_dim, z_channels, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Image -> latent, taking the distribution's mode rather than a sample."""
        posterior = DiagonalGaussianDistribution(self.quant_conv(self.encoder(x)))
        return posterior.mode()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Latent -> image."""
        return self.decoder(self.post_quant_conv(z))

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return z, self.decode(z), None