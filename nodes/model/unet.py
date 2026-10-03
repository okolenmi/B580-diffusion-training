"""The SDXL UNet, reimplemented rather than imported from ComfyUI.

Design doc 12, section 7.3, section A. `TimestepBlock`,
`TimestepEmbedSequential`, `Upsample`, `Downsample`, `ResBlock` and
`UNetModel` -- the published SDXL diffusion U-Net, which is the
ADM/guided-diffusion architecture with Stable Diffusion's conditioning
attached.

**The contract is the checkpoint.** 1,680 tensors with fixed names:
`time_embed.0`/`time_embed.2`, `label_emb.0.0`/`label_emb.0.2`, then
`input_blocks.N.M...`, `middle_block.M...`, `output_blocks.N.M...`, `out.0`/
`out.2`. Every one of those names is a key in the `.safetensors` this loads,
and this project's LoRA block-weight paths are written in terms of them
(`input_blocks.3.1.transformer_blocks.0.attn1.to_q`). So the names, the
order of the `nn.Sequential`s inside each block, and the shapes are all
fixed -- including one that looks like a mistake:

**`label_emb` is wrapped in a redundant `nn.Sequential`.** ComfyUI builds

    nn.Sequential(nn.Sequential(Linear, SiLU, Linear))

so the keys are `label_emb.0.0.weight`, not `label_emb.0.weight`. That
nesting is in the checkpoint; unwrapping it would leave 8 of the UNet's
tensors unloaded and `load_state_dict(strict=False)` would report them as
missing and carry on. It is reproduced deliberately and `load_state_dict`
is checked against a real checkpoint rather than trusted.

**What is not ported**, all of it either unused by SDXL or ComfyUI plumbing:

* `VideoResBlock`, `SpatialVideoTransformer`, `use_temporal_resblock`,
  `use_temporal_attention`, `time_context_dim`, `merge_strategy`,
  `merge_factor`, `video_kernel_size`, `use_spatial_context` -- video.
* `control` and `apply_control` -- ComfyUI's ControlNet hook. This project
  does not use ControlNet.
* The whole `transformer_options` patch protocol: `transformer_patches`,
  `input_block_patch`, `middle_block_after_patch`, `output_block_patch`,
  `emb_patch`, `forward_timestep_embed_patch`, `block`, `block_index`,
  `transformer_index`, `activations_shape`, `original_shape`, plus the
  `WrapperExecutor` indirection in `UNetModel.forward`.
* `operations.conv_nd` / `GroupNorm` / `Linear`, and `use_new_attention_order`
  and `num_heads_upsample`, which are dead in ComfyUI too -- the first is
  declared and never read, the second is documented as deprecated.
* `legacy`, `num_attention_blocks`, `disable_self_attentions`,
  `disable_middle_self_attn`, `n_embed`/`id_predictor`, `heatmap_head`,
  `use_scale_shift_norm`, `ResBlock`'s `use_conv`/`skip_t_emb`/
  `exchange_temb_dims`, and `Upsample`/`Downsample`'s `dims != 2` path.
  SDXL builds none of them. `legacy` in particular only ever changed how
  `dim_head` was derived, and SDXL passes `legacy: false`.

**Two details kept because they are load-bearing and not obvious:**

* `output_shape` is threaded into `Upsample`, so each upsample lands on the
  skip connection's spatial size rather than on exactly double. Without it a
  latent size that is not a power of two produces a skip that cannot be
  concatenated.
* `ResBlock.forward` calls `checkpoint(self._forward, ...)`, and
  `_forward` is a *bound method*. `block_profiler.py` reads
  `run_function.__self__` to label the block, and it is the real module
  instance only because it is bound. An inline forward would make every
  ResBlock label fall back to the opaque form.

Provenance: follows ComfyUI's
`comfy/ldm/modules/diffusionmodules/openaimodel.py` (Apache-2.0), which
follows the published architecture.
"""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SpatialTransformer
from .checkpoint import checkpoint
from .timestep_embedding import timestep_embedding

__all__ = [
    "Downsample",
    "ResBlock",
    "TimestepBlock",
    "TimestepEmbedSequential",
    "UNetModel",
    "Upsample",
]


class TimestepBlock(nn.Module):
    """A module whose forward takes timestep embeddings as a second argument."""

    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Runs its children, giving the ones that take embeddings the embeddings.

    The dispatch is by type, which is why the children are kept in one flat
    sequence rather than a tree: a `ResBlock` needs `(x, emb)`, a
    `SpatialTransformer` needs `(x, context)`, and everything else needs
    `x` alone.
    """

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
    """Nearest-neighbour upsampling by two, with an optional convolution."""

    def __init__(self, channels: int, use_conv: bool,
                 out_channels: int | None = None, padding: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, 3,
                                  padding=padding)

    def forward(self, x, output_shape=None):
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, "
                             f"got {x.shape[1]}")
        if output_shape is None:
            shape = [x.shape[2] * 2, x.shape[3] * 2]
        else:
            # Match the skip connection's spatial size rather than doubling.
            # At a latent size that is not a power of two these differ, and
            # the skip concatenation later in UNetModel fails on the mismatch.
            shape = [output_shape[2], output_shape[3]]
        x = F.interpolate(x, size=shape, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Halve the spatial size, with a strided convolution or average pooling."""

    def __init__(self, channels: int, use_conv: bool,
                 out_channels: int | None = None, padding: int = 1) -> None:
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.op = nn.Conv2d(self.channels, self.out_channels, 3,
                                stride=2, padding=padding)
        else:
            if self.channels != self.out_channels:
                raise ValueError("average pooling cannot change the channel "
                                 "count, so out_channels must match")
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if x.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, "
                             f"got {x.shape[1]}")
        return self.op(x)


class ResBlock(TimestepBlock):
    """A residual block conditioned on the timestep embedding.

    GroupNorm -> SiLU -> conv, add the projected embedding, GroupNorm -> SiLU
    -> dropout -> conv, add the skip. The embedding is *additive*: SDXL does
    not use the scale-and-shift (FiLM) form, which ComfyUI keeps behind
    `use_scale_shift_norm`.
    """

    def __init__(self, channels: int, emb_channels: int, dropout: float,
                 out_channels: int | None = None, use_checkpoint: bool = False,
                 up: bool = False, down: bool = False) -> None:
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        # `self._forward` is a bound method on purpose: block_profiler.py
        # reads run_function.__self__ to label the block, and only a bound
        # method carries the module instance.
        return checkpoint(self._forward, (x, emb), self.parameters(),
                          self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            # Resample before the convolution rather than after, so the
            # 3x3 kernel runs at the lower resolution.
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = self.h_upd(in_rest(x))
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    """The SDXL UNet: in `latent + timestep + context + y`, out a latent.

    :param in_channels: channels in the input latent (4 for SDXL).
    :param model_channels: base channel count; every level is a multiple.
    :param out_channels: channels in the output (4 for SDXL).
    :param num_res_blocks: residual blocks per level, as a list matching
        `channel_mult`.
    :param channel_mult: channel multiplier per resolution level.
    :param num_head_channels: fixed channel width per attention head; heads
        are derived as `channels // num_head_channels`.
    :param context_dim: width of the cross-attention context (2048 for SDXL,
        which is the CLIP-L pooled output concatenated with SDXL's time
        embedding).
    :param adm_in_channels: width of the additional-conditioning vector `y`.
    :param transformer_depth: attention blocks per input level.
    :param transformer_depth_middle: attention blocks in the middle. -1
        leaves out the middle block entirely.
    :param transformer_depth_output: attention blocks per output level.
    :param use_linear_in_transformer: use `Linear` rather than `Conv2d` for
        the spatial transformer's projections. SDXL uses True.
    :param num_classes: None, an int (class embedding), "continuous", or
        "sequential" (the ADM conditioning vector SDXL uses).
    :param use_checkpoint: activation checkpoint every ResBlock and spatial
        transformer.
    """

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks,
        dropout: float = 0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample: bool = True,
        num_classes=None,
        use_checkpoint: bool = False,
        num_head_channels: int = -1,
        use_spatial_transformer: bool = False,
        transformer_depth=1,
        context_dim=None,
        adm_in_channels=None,
        transformer_depth_middle=None,
        transformer_depth_output=None,
        use_linear_in_transformer: bool = False,
    ) -> None:
        super().__init__()

        if context_dim is not None and not use_spatial_transformer:
            raise ValueError(
                "context_dim is set, so use_spatial_transformer must be too: "
                "there is nowhere else for the cross-attention conditioning "
                "to go")
        if num_head_channels == -1:
            raise ValueError("num_head_channels must be set")

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "num_res_blocks must be an int or a list as long as "
                    f"channel_mult: {len(num_res_blocks)} vs "
                    f"{len(channel_mult)}")
            self.num_res_blocks = list(num_res_blocks)

        # Both lists are consumed by pop() as the blocks are built, so they
        # are copied: the caller's config must survive a second UNet.
        transformer_depth = list(transformer_depth)
        transformer_depth_output = list(transformer_depth_output)

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes == "sequential":
            if adm_in_channels is None:
                raise ValueError(
                    "num_classes='sequential' needs adm_in_channels")
            # The inner nn.Sequential is redundant and kept on purpose: the
            # checkpoint's keys are label_emb.0.0.* and label_emb.0.2.*, and
            # unwrapping it renames all four tensors.
            self.label_emb = nn.Sequential(
                nn.Sequential(
                    nn.Linear(adm_in_channels, time_embed_dim),
                    nn.SiLU(),
                    nn.Linear(time_embed_dim, time_embed_dim),
                )
            )
        elif isinstance(self.num_classes, int):
            self.label_emb = nn.Embedding(self.num_classes, time_embed_dim)
        elif self.num_classes == "continuous":
            self.label_emb = nn.Linear(1, time_embed_dim)
        elif self.num_classes is not None:
            raise ValueError(f"unrecognised num_classes: "
                             f"{self.num_classes!r}")

        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels

        def heads_for(channels: int) -> tuple[int, int]:
            """(heads, dim_head) for a level, from the fixed head width.

            SDXL's config fixes `num_head_channels` rather than the head
            count, and this is what makes `inner_dim == channels` inside
            every SpatialTransformer -- the coincidence that hides ComfyUI's
            `proj_out` bug, see nodes/model/attention.py.
            """
            n_heads = channels // num_head_channels
            return n_heads, num_head_channels

        def attention_layer(channels: int, depth: int):
            n_heads, dim_head = heads_for(channels)
            return SpatialTransformer(
                channels, n_heads, dim_head, depth=depth,
                context_dim=context_dim,
                use_linear=use_linear_in_transformer,
            )

        def resblock(channels: int, out_channels: int | None, *,
                     down: bool = False, up: bool = False):
            return ResBlock(
                channels=channels,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_channels,
                use_checkpoint=use_checkpoint,
                down=down,
                up=up,
            )

        # -- down ---------------------------------------------------------
        for level, mult in enumerate(channel_mult):
            for _ in range(self.num_res_blocks[level]):
                layers = [resblock(ch, mult * model_channels)]
                ch = mult * model_channels
                depth = transformer_depth.pop(0)
                if depth > 0:
                    layers.append(attention_layer(ch, depth))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                out_ch = ch
                # ComfyUI chooses between a down-ResBlock and a plain
                # Downsample here on `resblock_updown`, which SDXL leaves
                # false. Only the Downsample branch is carried over.
                self.input_blocks.append(TimestepEmbedSequential(
                    Downsample(ch, conv_resample, out_channels=out_ch)))
                ch = out_ch
                input_block_chans.append(ch)

        # -- middle -------------------------------------------------------
        if transformer_depth_middle is not None and \
                transformer_depth_middle >= 0:
            n_heads, dim_head = heads_for(ch)
            middle = [
                resblock(ch, None),
                SpatialTransformer(ch, n_heads, dim_head,
                                   depth=transformer_depth_middle,
                                   context_dim=context_dim,
                                   use_linear=use_linear_in_transformer),
                resblock(ch, None),
            ]
            self.middle_block = TimestepEmbedSequential(*middle)
        else:
            self.middle_block = None

        # -- up -----------------------------------------------------------
        self.output_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                skip_channels = input_block_chans.pop()
                layers = [resblock(ch + skip_channels, model_channels * mult)]
                ch = model_channels * mult
                depth = transformer_depth_output.pop()
                if depth > 0:
                    layers.append(attention_layer(ch, depth))
                if level and i == self.num_res_blocks[level]:
                    layers.append(
                        Upsample(ch, conv_resample, out_channels=ch))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, timesteps, context=None, y=None):
        """
        :param x: `[N, in_channels, H, W]` latents.
        :param timesteps: 1-D batch of timesteps, which may be fractional.
        :param context: `[N, tokens, context_dim]` cross-attention
            conditioning.
        :param y: `[N, adm_in_channels]` additional conditioning. Required
            exactly when the model is class-conditional.
        """
        if (y is not None) != (self.num_classes is not None):
            raise ValueError(
                f"y must be given if and only if the model is "
                f"class-conditional: num_classes={self.num_classes!r}, "
                f"y={'given' if y is not None else 'not given'}")

        skips = []
        t_emb = timestep_embedding(timesteps, self.model_channels).to(x.dtype)
        emb = self.time_embed(t_emb)
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            skips.append(h)

        if self.middle_block is not None:
            h = self.middle_block(h, emb, context)

        for module in self.output_blocks:
            # The skip is the matching input block, in reverse order. It is
            # popped rather than indexed so that a shape mismatch shows up as
            # an IndexError instead of a wrong-but-plausible tensor.
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, emb, context,
                       output_shape=skips[-1].shape if skips else None)
        return self.out(h.type(x.dtype))