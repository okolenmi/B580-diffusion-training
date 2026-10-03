"""The attention stack, reimplemented rather than imported from ComfyUI.

Design doc 12, section 7.3, section A. These are `CrossAttention`,
`FeedForward`, `GEGLU`, `BasicTransformerBlock` and `SpatialTransformer` --
everything between the UNet's ResBlocks and the raw tensors.

**What is fixed rather than matched.** ComfyUI's `BasicTransformerBlock`
(1,335-line `attention.py`) is one class serving SDXL, SD3, Flux, Hunyuan
and Stable Cascade from ComfyUI's own in-memory checkpoint format, so
almost all of it is a model-patching framework rather than a transformer:
`transformer_patches`, `transformer_patches_replace`, `attn1_patch`,
`attn2_patch`, `middle_patch`, `attn1_output_patch`, `block`, `block_index`,
`extra_options`, `switch_temporal_ca_to_sa`, `disable_temporal_crossattn`.
Roughly 15 lines of that forward are arithmetic. Ours is those 15 lines, and
`attention_checkpointing.py`'s patch -- which wraps `forward` -- becomes
proportionally smaller because it patches a small function.

Three more things not carried over:

* **`operations.Linear` / `GroupNorm` / `LayerNorm`.** ComfyUI's
  dtype/device dispatch wrappers, threaded through every constructor as
  `dtype=`/`device=`/`operations=`. This project passes device and dtype
  explicitly and never through them.
* **`transformer_options`.** Present on nearly every forward for patching.
  Ours takes `context` and nothing else.
* **`SpatialTransformer`'s `is_linear` branch has a bug, and this project
  uses that branch.** ComfyUI builds `proj_out` as
  `Linear(in_channels, inner_dim)` -- identical to `proj_in`, with the in and
  out roles never swapped, so the branch cannot work for any model where
  `inner_dim != in_channels`. It is invisible for SDXL *by coincidence*:
  `inner_dim` is `num_heads * dim_head` with `dim_head = num_head_channels
  = 64` and `num_heads = ch // 64`, so `inner_dim == ch == in_channels` at
  every level, and both projections come out as e.g. `Linear(640, 640)`.
  Verified by constructing the real UNet: 11 SpatialTransformers, every one
  square. Ours is written `Linear(inner_dim, in_channels)`, which is the
  same tensor for SDXL and correct for anything else.

**The contract that does have to match** is the parameter names and shapes,
because they are what the checkpoint's keys and this project's LoRA target
paths bind to: `attn1`/`attn2` with `to_q`, `to_k`, `to_v`, `to_out.0`;
`ff` as `net.0`/`net.2`; `norm1`/`norm2`/`norm3`; and, on the
SpatialTransformer, `norm`, `proj_in`, `transformer_blocks.N`,
`proj_out`. Verified: constructing both this and ComfyUI's for every SDXL
transformer level gives an identical `state_dict()` -- same keys, same
shapes, 46 / 206 / 206 parameters at depths 2 / 10 / 10. And with weights
supplied, the forwards are **bitwise identical** on CPU across both
projection branches and every block type (see
`smoke_test_attention.py`).

**Two more ComfyUI quirks found while porting, neither copied.**

* `BasicTransformerBlock.__init__` takes `inner_dim` as a *parameter
  defaulting to None*, and `ff_in` is `ff_in or inner_dim is not None`.
  Reading it as `inner_dim = n_heads * d_head` and keeping that expression
  looks equivalent and is not: it makes `ff_in` permanently true and adds a
  `norm_in` and an `ff_in` FeedForward whose weights are in no checkpoint.
  Checked against the real UNet -- 70 blocks, none with `norm_in`, zero
  `state_dict` keys mentioning `ff_in` -- so the parameter version is what
  SDXL actually builds.
* `comfy.ops.Linear` leaves its weight **uninitialised**. Constructing one
  and reading it gives values around 3e29, because ComfyUI assumes a
  checkpoint always overwrites them. Harmless there and a trap here: the
  first comparison run produced an all-NaN forward on *both* sides, from
  the weights rather than the arithmetic. Ours uses PyTorch's default init.
  Anything comparing the two has to supply weights from one generator.

Provenance: follows ComfyUI's `comfy/ldm/modules/attention.py` (Apache-2.0)
for the arithmetic, which is the published transformer. The dispatch layer is
not carried over.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "BasicTransformerBlock",
    "CrossAttention",
    "FeedForward",
    "GEGLU",
    "SpatialTransformer",
    "group_norm_32",
]


def _or(value, fallback):
    return fallback if value is None else value


def group_norm_32(channels: int) -> nn.GroupNorm:
    """32-group GroupNorm, eps 1e-6, affine -- what SDXL normalises with.

    The name says what it is. ComfyUI calls this `Normalize`, which reads as
    a generic helper and is in fact a specific configuration.
    """
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6,
                        affine=True)


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               heads: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Scaled dot-product attention over `heads` heads, shapes unchanged.

    `q`, `k`, `v` are `[batch, seq, heads * dim_head]`; the head split takes
    the *outer* stride of the last dimension, which is the convention the
    checkpoint's projection weights were trained under and the one ComfyUI's
    `attention_pytorch` uses. Getting it wrong transposes nothing and
    changes nothing about the shapes -- so it is not detectable by a shape
    test, which is why the characterisation tests compare numbers.

    `mask` is added for CLIP, whose text transformer is causal. It arrives as
    `[seq, seq]` and is broadcast to `[batch, 1, seq, seq]` here, which is
    the same broadcast ComfyUI performs on the way in; doing it in one place
    keeps the head split in one place too.

    `scaled_dot_product_attention` rather than a hand-rolled softmax, so the
    backend picks its own fused kernel per device. On a 12 GB Intel card that
    is the difference between an attention that fits and one that does not.
    """
    batch, q_len, width = q.shape
    head_dim = width // heads
    shape = (batch, -1, heads, head_dim)

    def split(t):
        return t.view(*shape).transpose(1, 2)

    if mask is not None and mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    out = F.scaled_dot_product_attention(split(q), split(k), split(v),
                                         attn_mask=mask, dropout_p=0.0,
                                         is_causal=False)
    return out.transpose(1, 2).reshape(batch, q_len, width)


class GEGLU(nn.Module):
    """Gated GELU: one projection to twice the width, split, gate the halves."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    """The two-layer MLP between attention blocks.

    `net.0` and `net.2` are part of the checkpoint's key names, so the
    `nn.Sequential` with a `Dropout` at index 1 is not incidental.
    """

    def __init__(self, dim: int, dim_out: int | None = None, mult: int = 4,
                 glu: bool = False, dropout: float = 0.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = _or(dim_out, dim)
        project_in = (
            GEGLU(dim, inner_dim) if glu
            else nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
        )
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """Multi-head attention over a context tensor.

    `attn1` in a `BasicTransformerBlock` is the self-attention case and is
    built with `context_dim=None`; `attn2` is cross-attention. The parameter
    names `to_q`/`to_k`/`to_v`/`to_out.0` are the checkpoint's.
    """

    def __init__(self, query_dim: int, heads: int, dim_head: int,
                 dropout: float = 0.0, context_dim: int | None = None) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        context_dim = _or(context_dim, query_dim)

        self.heads = heads
        self.query_dim = query_dim
        self.context_dim = context_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim),
                                    nn.Dropout(dropout))

    def forward(self, x, context=None, value=None):
        if context is None:
            # Falling back to self-attention is only meaningful when the
            # context and query widths agree. When they differ, substituting
            # `x` produces a matmul error about shapes with no hint as to
            # the cause -- which is what ComfyUI does here. Say it instead.
            if self.context_dim != self.query_dim:
                raise ValueError(
                    f"no context given, but this is a cross-attention over "
                    f"{self.context_dim} channels attending to "
                    f"{self.query_dim}: it cannot attend to itself")
            context = x
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context if value is None else value)
        return self.to_out(_attention(q, k, v, self.heads))


class BasicTransformerBlock(nn.Module):
    """Self-attention, cross-attention, feed-forward, with residuals.

    ComfyUI's version also takes `disable_self_attn`,
    `disable_temporal_crossattention`, `switch_temporal_ca_to_sa` and
    `attn_precision`. None is carried over: the first three are for the
    video and image-only variants it multiplexes through this class, the
    last is a precision knob this project does not expose, and SDXL builds
    every block with all of them off.

    `attention_checkpointing.py` replaces `forward` on this class at runtime,
    which is why its shape matters to that file.
    """

    def __init__(self, dim: int, n_heads: int, d_head: int,
                 dropout: float = 0.0, context_dim: int | None = None,
                 gated_ff: bool = True, ff_in: bool = False,
                 inner_dim: int | None = None) -> None:
        super().__init__()
        # `inner_dim` is a parameter defaulting to None, not something
        # computed from the head count. That matters: `ff_in` is
        # `ff_in or inner_dim is not None`, so computing an inner_dim
        # locally would make this *always* true and add a `norm_in` and an
        # `ff_in` FeedForward that the checkpoint has no weights for. Copied
        # as `inner_dim = n_heads * d_head` it looks equivalent and is not.
        self.has_ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        self.is_res = inner_dim == dim

        if self.has_ff_in:
            self.norm_in = nn.LayerNorm(dim)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout,
                                     glu=gated_ff)

        self.attn1 = CrossAttention(query_dim=inner_dim, heads=n_heads,
                                    dim_head=d_head, dropout=dropout,
                                    context_dim=None)
        self.attn2 = CrossAttention(query_dim=inner_dim, heads=n_heads,
                                    dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim)
        self.norm1 = nn.LayerNorm(inner_dim)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout,
                              glu=gated_ff)

    def forward(self, x, context=None, transformer_options=None):
        if self.has_ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            # Guarded on is_res, so with inner_dim != dim the feed-forward
            # *replaces* the width rather than adding to it.
            if self.is_res:
                x = x + x_skip

        x = self.attn1(self.norm1(x), context=None) + x
        if self.attn2 is not None:
            x = self.attn2(self.norm2(x), context=context) + x

        if self.is_res:
            x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x = x + x_skip
        return x


class SpatialTransformer(nn.Module):
    """`[batch, channels, h, w]` in and out, attention over the h*w axis.

    Normalises, projects to `inner_dim`, attends with `depth` blocks, and
    projects back with a residual against the input. `context` is either one
    tensor shared by every block or one per block.
    """

    def __init__(self, in_channels: int, n_heads: int, d_head: int, depth: int,
                 dropout: float = 0.0, context_dim: int | None = None,
                 use_linear: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.use_linear = use_linear
        inner_dim = n_heads * d_head
        self.norm = group_norm_32(in_channels)

        # ComfyUI has `Linear(in_channels, inner_dim)` for *both* projections.
        # Same tensor for SDXL (inner_dim == in_channels), wrong for anything
        # else -- see the module docstring.
        self.proj_in = (nn.Linear(in_channels, inner_dim) if use_linear
                        else nn.Conv2d(in_channels, inner_dim, kernel_size=1))
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout,
                                  context_dim=context_dim)
            for _ in range(depth)
        ])
        self.proj_out = (nn.Linear(inner_dim, in_channels) if use_linear
                         else nn.Conv2d(inner_dim, in_channels,
                                        kernel_size=1))

    def forward(self, x, context=None, transformer_options=None):
        if isinstance(context, list):
            if len(context) != len(self.transformer_blocks):
                raise ValueError(
                    f"expected {len(self.transformer_blocks)} context "
                    f"tensors, got {len(context)}")
            blocks = zip(self.transformer_blocks, context)
        else:
            blocks = ((block, context)
                      for block in self.transformer_blocks)

        batch, _, height, width = x.shape
        x_in = x
        x = self.norm(x)

        # The two branches project on opposite sides of the flattening, and
        # it matters. With `use_linear` the projection is applied to the
        # flattened [batch, h*w, channels] tensor, so it acts on the channel
        # axis; applying it before the flatten would act on the *width* axis
        # instead, which is a shape error rather than a wrong number -- it
        # still fails loudly, which is the only reason it was caught quickly.
        # The conv branch is the mirror image: conv needs the spatial axes,
        # so it runs before the flatten, and its `proj_out` runs after the
        # reshape back.
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.movedim(1, 3).flatten(1, 2).contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for block, ctx in blocks:
            x = block(x, context=ctx)

        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(batch, height, width, -1).movedim(3, 1).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in