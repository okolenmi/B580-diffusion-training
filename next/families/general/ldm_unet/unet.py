"""The UNet of latent diffusion. Key names (``input_blocks``, ``middle_block``, ``output_blocks``, ``out``,
``time_embed``, ``label_emb``) match the published checkpoints.

Gradient checkpointing is not implemented here. ``UNetModel.set_checkpointer`` accepts a callable
``checkpointer(fn, inputs, params)`` that residual and transformer blocks use to run their forward; ``None`` runs it directly.
The training runtime supplies the callable, so this module stays free of runtime concerns.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

from ..embeddings import timestep_embedding
from .attention import BasicTransformerBlock, SpatialTransformer

Checkpointer = Callable[[Callable[..., Any], tuple, Iterable[nn.Parameter]], Any]


class TimestepBlock(nn.Module):
    """Marks layers whose forward takes the timestep embedding."""

    def forward(self, x, emb):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None, output_shape=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            elif isinstance(layer, Upsample):
                x = layer(x, output_shape=output_shape)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(channels, self.out_channels, 3, padding=padding)

    def forward(self, x, output_shape=None):
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[1]}")
        size = [x.shape[2] * 2, x.shape[3] * 2] if output_shape is None else [output_shape[2], output_shape[3]]
        x = F.interpolate(x, size=size, mode="nearest")
        return self.conv(x) if self.use_conv else x


class Downsample(nn.Module):
    def __init__(self, channels: int, use_conv: bool, out_channels: int | None = None, padding: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        if use_conv:
            self.op = nn.Conv2d(channels, self.out_channels, 3, stride=2, padding=padding)
        else:
            if channels != self.out_channels:
                raise ValueError("average pooling cannot change the channel count")
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {x.shape[1]}")
        return self.op(x)


class ResBlock(TimestepBlock):
    def __init__(self, channels: int, emb_channels: int, dropout: float, out_channels: int | None = None,
                 up: bool = False, down: bool = False) -> None:
        super().__init__()
        self.out_channels = out_channels or channels
        self.checkpointer: Checkpointer | None = None
        self.in_layers = nn.Sequential(nn.GroupNorm(32, channels), nn.SiLU(),
                                       nn.Conv2d(channels, self.out_channels, 3, padding=1))
        self.updown = up or down
        if up:
            self.h_upd, self.x_upd = Upsample(channels, False), Upsample(channels, False)
        elif down:
            self.h_upd, self.x_upd = Downsample(channels, False), Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, self.out_channels))
        self.out_layers = nn.Sequential(nn.GroupNorm(32, self.out_channels), nn.SiLU(), nn.Dropout(p=dropout),
                                        nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))
        self.skip_connection = (nn.Identity() if self.out_channels == channels
                                else nn.Conv2d(channels, self.out_channels, 1))

    def forward(self, x, emb):
        if self.checkpointer is None:
            return self._forward(x, emb)
        return self.checkpointer(self._forward, (x, emb), self.parameters())

    def _forward(self, x, emb):
        if self.updown:
            rest, conv = self.in_layers[:-1], self.in_layers[-1]
            h = conv(self.h_upd(rest(x)))
            x = self.x_upd(x)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while emb_out.ndim < h.ndim:
            emb_out = emb_out[..., None]
        return self.skip_connection(x) + self.out_layers(h + emb_out)


class UNetModel(nn.Module):
    def __init__(self, *, in_channels: int, model_channels: int, out_channels: int, num_res_blocks,
                 channel_mult=(1, 2, 4, 8), dropout: float = 0.0, conv_resample: bool = True, num_classes=None,
                 num_head_channels: int, transformer_depth, context_dim: int | None = None,
                 adm_in_channels: int | None = None, transformer_depth_middle: int | None = None,
                 transformer_depth_output, use_linear_in_transformer: bool = False) -> None:
        super().__init__()
        self.model_channels, self.num_classes = model_channels, num_classes
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        elif len(num_res_blocks) == len(channel_mult):
            self.num_res_blocks = list(num_res_blocks)
        else:
            raise ValueError(f"num_res_blocks has {len(num_res_blocks)} entries for {len(channel_mult)} levels")
        depth_in, depth_out = list(transformer_depth), list(transformer_depth_output)
        embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        if num_classes == "sequential":
            if adm_in_channels is None:
                raise ValueError("num_classes='sequential' needs adm_in_channels")
            self.label_emb = nn.Sequential(nn.Sequential(nn.Linear(adm_in_channels, embed_dim), nn.SiLU(),
                                                         nn.Linear(embed_dim, embed_dim)))
        elif isinstance(num_classes, int):
            self.label_emb = nn.Embedding(num_classes, embed_dim)
        elif num_classes is not None:
            raise ValueError(f"unrecognised num_classes {num_classes!r}")

        def attention(channels: int, depth: int) -> SpatialTransformer:
            return SpatialTransformer(channels, channels // num_head_channels, num_head_channels, depth,
                                      context_dim=context_dim, use_linear=use_linear_in_transformer)

        def res(channels: int, out: int | None) -> ResBlock:
            return ResBlock(channels, embed_dim, dropout, out)

        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(nn.Conv2d(in_channels, model_channels, 3, padding=1))])
        skip_channels, ch = [model_channels], model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [res(ch, mult * model_channels)]
                ch = mult * model_channels
                depth = depth_in.pop(0)
                if depth > 0:
                    layers.append(attention(ch, depth))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                skip_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, conv_resample, out_channels=ch)))
                skip_channels.append(ch)

        self.middle_block = None
        if transformer_depth_middle is not None and transformer_depth_middle >= 0:
            self.middle_block = TimestepEmbedSequential(res(ch, None), attention(ch, transformer_depth_middle), res(ch, None))

        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                layers = [res(ch + skip_channels.pop(), model_channels * mult)]
                ch = model_channels * mult
                depth = depth_out.pop()
                if depth > 0:
                    layers.append(attention(ch, depth))
                if level and i == self.num_res_blocks[level]:
                    layers.append(Upsample(ch, conv_resample, out_channels=ch))
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        self.out = nn.Sequential(nn.GroupNorm(32, ch), nn.SiLU(), nn.Conv2d(model_channels, out_channels, 3, padding=1))

    def set_checkpointer(self, checkpointer: Checkpointer | None, attention_fraction: float = 1.0) -> None:
        """Residual blocks and a share of the transformer blocks run through ``checkpointer`` (``None`` turns it off).

        ``attention_fraction`` in [0, 1] picks that share, spread evenly through the network: 1 checkpoints every
        transformer block, 0.5 every other one, 0 none.
        """
        if not 0.0 <= attention_fraction <= 1.0:
            raise ValueError(f"attention_fraction must be in [0, 1], got {attention_fraction}")
        blocks = [m for m in self.modules() if isinstance(m, BasicTransformerBlock)]
        for module in self.modules():
            if isinstance(module, ResBlock):
                module.checkpointer = checkpointer
        for index, block in enumerate(blocks):
            chosen = int((index + 1) * attention_fraction) > int(index * attention_fraction)
            block.checkpointer = checkpointer if chosen else None

    def forward(self, x, timesteps, context=None, y=None):
        if (y is not None) != (self.num_classes is not None):
            raise ValueError(f"y must be given exactly when the model is class-conditional (num_classes={self.num_classes!r})")
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels).to(x.dtype))
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
        skips = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            skips.append(h)
        if self.middle_block is not None:
            h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, emb, context, output_shape=skips[-1].shape if skips else None)
        return self.out(h.type(x.dtype))
