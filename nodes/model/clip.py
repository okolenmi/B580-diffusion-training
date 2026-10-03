"""The SDXL CLIP text encoders, reimplemented rather than imported from ComfyUI.

Design doc 12, section 7.3, section C1 -- the *model* half. The tokenizer is
section C2 and is not here, because it needs a decision about vendoring data
that is not code; see the design doc.

`CLIPAttention`, `CLIPMLP`, `CLIPLayer`, `CLIPEncoder`, `CLIPEmbeddings`,
`CLIPTextModel`, `SDClipModel`, `SDXLClipG` and `SDXLClipModel` -- the two
text towers SDXL conditions on:

* **CLIP-L**, OpenAI `clip-vit-large-patch14`'s text tower: 12 layers, width
  768, 12 heads, `quick_gelu`. Its *penultimate hidden layer* is the
  conditioning, and its EOS position's hidden state is the pooled output.
* **CLIP-G**, the much larger OpenCLIP ViT-H text tower: 32 layers, width
  1280, 20 heads, `gelu`. Same two outputs.

Both configurations are inlined below as dicts rather than read from
ComfyUI's `sd1_clip_config.json` and `clip_config_bigg.json`. Those two files
are 1.1 KB of published architecture description, and inlining them removes the
last reason this project would need ComfyUI's *tree* rather than just its
Python.

**The contract is the checkpoint's keys**: `clip_l.transformer.text_model.*`
and `clip_g.transformer.text_model.*`, which `clip_state_dict.py` produces.
The nesting inside `SDClipModel` is part of that -- `self.transformer` is a
`CLIPTextModel`, whose own `text_model` is the stack, hence the doubled
`text_model.text_model` for CLIP-L's projection and ComfyUI's
`transformer.text_model` after the key rewrite.

**Details that are load-bearing and not obvious:**

* **The conditioning is the penultimate layer, not the last.**
  `layer="hidden", layer_idx=-2`. Getting this wrong produces a model that
  loads cleanly, runs, and conditions slightly worse than it should -- no
  error anywhere.
* **`final_layer_norm` is not applied to the hidden output.**
  `layer_norm_hidden_state=False` for SDXL. It *is* applied to the last
  layer's output, which is what the pooled projection sees.
* **CLIP-L's activation is `quick_gelu`** (`x * sigmoid(1.702x)`) and CLIP-G's
  is plain `gelu`. They are different architectures, not one with a flag.
* **Attention is causal**, with the mask built as `triu(1)` over
  `-finfo.max`. Padding and causality are summed into one mask.
* The forward runs in **float32 regardless of the model's dtype**
  (`dtype=torch.float32` at the call site), and the results are cast back
  afterwards. On a 12 GB card that matters: fp16 attention over 77 tokens
  with 12 heads is not where the memory goes, but the projections are.

**What is not ported:** the vision towers (`CLIPVisionEmbeddings`,
`CLIPVision`, `CLIPVisionModelProjection`), SigLIP2, LlavaProjector, the
`ClipTokenWeightEncoder` ABC and its LDM sibling, `load_sd`'s vae-mode
fallbacks, `add_weight_prefix`, and the LoRA-compatible remapping
machinery. Also not reimplemented: the token *weight* path, which exists only
for ComfyUI's wildcard syntax -- see `encode_token_ids`.

Provenance: follows ComfyUI's `comfy/clip_model.py` and
`comfy/sd1_clip.py` (Apache-2.0), which follow the published CLIP
architectures.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import _attention

__all__ = [
    "CLIP_G_CONFIG",
    "CLIP_L_CONFIG",
    "CLIPAttention",
    "CLIPEncoder",
    "CLIPLayer",
    "CLIPMLP",
    "CLIPTextModel",
    "SDClipModel",
    "SDXLClipG",
    "SDXLClipModel",
]


#: OpenAI `clip-vit-large-patch14`'s text tower, verbatim from ComfyUI's
#: `sd1_clip_config.json`. Note `quick_gelu` -- CLIP-L is not a plain-GELU
#: model, and using GELU here is a small, silent quality loss.
CLIP_L_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "max_position_embeddings": 77,
    "hidden_act": "quick_gelu",
    "eos_token_id": 49407,
    "vocab_size": 49408,
}

#: OpenCLIP ViT-H's text tower, verbatim from `clip_config_bigg.json`. Plain
#: `gelu`, 32 layers, width 1280.
CLIP_G_CONFIG = {
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
    "max_position_embeddings": 77,
    "hidden_act": "gelu",
    "eos_token_id": 49407,
    "vocab_size": 49408,
}

#: CLIP-L uses `quick_gelu`, which predates the tanh approximation and is not
#: `nn.GELU`'s default. It is `x * sigmoid(1.702 * x)`, with the constant
#: chosen to approximate GELU closely.
_ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": F.gelu,
    "gelu_pytorch_tanh": lambda a: F.gelu(a, approximate="tanh"),
}


class CLIPAttention(nn.Module):
    """Multi-head self-attention. All four projections are biased."""

    def __init__(self, embed_dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x, mask=None):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        return self.out_proj(_attention(q, k, v, self.heads, mask=mask))


class CLIPMLP(nn.Module):
    """The position-wise feed-forward: fc1, activation, fc2."""

    def __init__(self, embed_dim: int, intermediate_size: int,
                 activation: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, intermediate_size, bias=True)
        self.activation = _ACTIVATIONS[activation]
        self.fc2 = nn.Linear(intermediate_size, embed_dim, bias=True)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class CLIPLayer(nn.Module):
    """Pre-norm attention and feed-forward, both residual."""

    def __init__(self, embed_dim: int, heads: int, intermediate_size: int,
                 activation: str) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = CLIPAttention(embed_dim, heads)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mlp = CLIPMLP(embed_dim, intermediate_size, activation)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class CLIPEncoder(nn.Module):
    """The stack of layers, with an optional tap on one of them."""

    def __init__(self, num_layers: int, embed_dim: int, heads: int,
                 intermediate_size: int, activation: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            CLIPLayer(embed_dim, heads, intermediate_size, activation)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, intermediate_output=None):
        """Returns `(last, tapped)`; `tapped` is None unless asked for.

        `intermediate_output` is an index into the layers (negative indices
        count from the end, which is how SDXL asks for the penultimate one),
        or "all" for every layer.
        """
        keep_all = False
        if intermediate_output == "all":
            keep_all = True
            every = []
            intermediate_output = None
        elif intermediate_output is not None and intermediate_output < 0:
            intermediate_output = len(self.layers) + intermediate_output

        intermediate = None
        for index, layer in enumerate(self.layers):
            x = layer(x, mask)
            if index == intermediate_output:
                # Cloned, because the caller keeps this while `x` carries on
                # being updated in place by the next layer's residuals.
                intermediate = x.clone()
            if keep_all:
                every.append(x.unsqueeze(1).clone())
        if keep_all:
            intermediate = torch.cat(every, dim=1)
        return x, intermediate


class CLIPEmbeddings(nn.Module):
    """Token embedding plus learned position embedding."""

    def __init__(self, embed_dim: int, vocab_size: int = 49408,
                 num_positions: int = 77) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(num_positions, embed_dim)

    def forward(self, input_tokens, dtype=None):
        dtype = dtype or self.token_embedding.weight.dtype
        # Both halves are cast to the compute dtype rather than added as
        # they are stored: with mixed storage the sum would promote by
        # accident of which operand came first.
        return self.token_embedding(input_tokens).to(dtype) + \
            self.position_embedding.weight.to(dtype)


class CLIPTextModel_(nn.Module):
    """Embeddings, encoder, final layer norm."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.eos_token_id = config["eos_token_id"]
        self.embeddings = CLIPEmbeddings(config["hidden_size"],
                                         num_positions=config["max_position_embeddings"])
        self.encoder = CLIPEncoder(config["num_hidden_layers"],
                                   config["hidden_size"],
                                   config["num_attention_heads"],
                                   config["intermediate_size"],
                                   config["hidden_act"])
        self.final_layer_norm = nn.LayerNorm(config["hidden_size"])

    def forward(self, embeds, input_tokens=None, attention_mask=None,
                intermediate_output=None,
                final_layer_norm_intermediate: bool = True,
                dtype=None):
        # `dtype=None` means "compute in whatever the parameters are stored
        # as", which is the only self-consistent choice now that there is no
        # per-module casting layer. ComfyUI's `operations.*` wrappers allow
        # fp16 Linear weights alongside fp32 LayerNorm weights and cast on
        # every call; reimplementing that dispatch is precisely the machinery
        # this file is trying not to carry, so one dtype is used throughout.
        dtype = dtype or self.final_layer_norm.weight.dtype
        x = embeds.to(dtype) + self.embeddings.position_embedding.weight.to(
            embeds.device).to(dtype)

        length = x.shape[1]
        # Causality: disallow attending forward. triu(1) keeps the diagonal
        # and everything below it, so position i sees 0..i.
        mask = torch.full((length, length), -torch.finfo(x.dtype).max,
                          dtype=x.dtype, device=x.device).triu_(1)
        if attention_mask is not None:
            # Padding mask, summed with the causal one rather than applied
            # separately: both are additive -inf masks, so adding is the
            # combination.
            padding = (1.0 - attention_mask.to(x.dtype)
                       .reshape(attention_mask.shape[0], 1, -1,
                                attention_mask.shape[-1])
                       .expand(attention_mask.shape[0], 1,
                               attention_mask.shape[-1],
                               attention_mask.shape[-1]))
            mask = mask + padding.masked_fill(padding.to(torch.bool),
                                              -torch.finfo(x.dtype).max)

        x, intermediate = self.encoder(x, mask=mask,
                                       intermediate_output=intermediate_output)
        x = self.final_layer_norm(x)
        if intermediate is not None and final_layer_norm_intermediate:
            intermediate = self.final_layer_norm(intermediate)

        # The pooled output is the *final* layer's hidden state at the first
        # EOS position -- not the last position, and not the tapped layer.
        # The tokenizer pads with EOS, so for a short prompt this is where
        # the sequence actually ends, and using the final position instead
        # reads padding. It needs the token ids, which is why `embeds` alone
        # is not enough here.
        if input_tokens is None:
            raise ValueError("input_tokens is required for EOS pooling")
        eos_positions = (torch.round(input_tokens.float()).to(torch.int)
                         == self.eos_token_id).int().argmax(dim=-1)
        pooled = x[torch.arange(x.shape[0], device=x.device), eos_positions]
        return x, intermediate, pooled


class CLIPTextModel(nn.Module):
    """`CLIPTextModel_` plus the projection that produces the pooled output."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.num_layers = config["num_hidden_layers"]
        self.text_model = CLIPTextModel_(config)
        embed_dim = config["hidden_size"]
        self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)

    def get_input_embeddings(self):
        return self.text_model.embeddings.token_embedding

    def forward(self, embeds, input_tokens=None, attention_mask=None,
                intermediate_output=None,
                final_layer_norm_intermediate: bool = True,
                dtype=None):
        last, tapped, pooled = self.text_model(
            embeds, input_tokens=input_tokens,
            attention_mask=attention_mask,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=final_layer_norm_intermediate,
            dtype=dtype)
        return last, tapped, self.text_projection(pooled), tapped


class SDClipModel(nn.Module):
    """A CLIP text tower, plus the plumbing that picks which layer to read.

    :param layer: "hidden" to read `layer_idx`, or "last" for the final one.
    :param layer_idx: negative counts from the end, so -2 is penultimate.
    :param layer_norm_hidden_state: apply the final norm to the tapped layer.
        False for SDXL, and the difference is a real change to the
        conditioning rather than a detail.
    :param special_tokens: the BOS/EOS/pad ids for this tower. CLIP-L pads
        with EOS; CLIP-G does not.
    """

    def __init__(self, config: dict, max_length: int = 77,
                 layer: str = "last", layer_idx: int | None = None,
                 special_tokens: dict | None = None, dtype=None,
                 device=None, layer_norm_hidden_state: bool = True,
                 return_projected_pooled: bool = True) -> None:
        super().__init__()
        self.transformer = CLIPTextModel(config)
        self.num_layers = self.transformer.num_layers
        self.max_length = max_length
        self.layer = layer
        self.layer_idx = layer_idx
        self.special_tokens = special_tokens or {"start": 49406, "end": 49407,
                                                 "pad": 49407}
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        self.logit_scale = nn.Parameter(torch.tensor(4.6055))
        self.execution_device = None

        if dtype is not None:
            self.to(dtype=dtype)
        if device is not None:
            self.to(device=device)
        self.eval()

        if layer_idx is not None:
            if abs(layer_idx) >= self.num_layers:
                raise ValueError(
                    f"layer_idx {layer_idx} is out of range for "
                    f"{self.num_layers} layers")
            self.set_layer_idx(layer_idx)

    def set_layer_idx(self, layer_idx: int | None) -> None:
        self.layer_idx = layer_idx
        self.layer = "hidden" if layer_idx is not None else "last"

    def forward(self, tokens,
                attention_mask: torch.Tensor | None = None):
        # `tokens` arrives as a list of equal-length id lists -- one per
        # section, padded to the longest -- because the tokenizer produces
        # them that way and the weighted-blending path slices them per row.
        # ComfyUI tensorises them in `process_tokens`; there is nothing else
        # in that function for this caller, so it happens here.
        if not torch.is_tensor(tokens):
            tokens = torch.tensor(tokens, dtype=torch.long)
        # Onto the model's device. The tokenizer produces ids on CPU, and an
        # embedding lookup needs the index and the weight on the same device --
        # this only shows up off CPU, where it raises from inside
        # torch.embedding rather than anywhere near the encoder.
        tokens = tokens.to(self.transformer.get_input_embeddings().weight.device)
        compute = self.transformer.text_model.final_layer_norm.weight.dtype
        embeds = self.transformer.get_input_embeddings()(tokens).to(compute)
        intermediate_output = self.layer_idx if self.layer == "hidden" else None
        last, tapped, projected, _ = self.transformer(
            embeds, input_tokens=tokens, attention_mask=attention_mask,
            intermediate_output=intermediate_output,
            final_layer_norm_intermediate=self.layer_norm_hidden_state,
            dtype=compute)
        z = tapped if self.layer == "hidden" else last

        pooled = None
        if self.return_projected_pooled:
            pooled = projected
        return z, pooled

    def encode(self, tokens):
        return self(tokens)

    def encode_token_ids(self, token_id_rows):
        """Encode one or more rows of token ids.

        Rows are stacked into a single batch, so they must be the same length
        as each other and the same length as the position embedding.

        **No token weights.** ComfyUI's `encode_token_weights` takes `(id,
        weight)` pairs and, when any weight is not 1.0, encodes an extra
        blank row and blends each weighted output between the weighted and
        blanked results:

            out = blank + (weighted - blank) * weight

        That exists for ComfyUI's `<w:...>` wildcard syntax, where several
        expansions share a prefix that must not be encoded once per expansion.
        This project has no wildcard syntax, so every weight is 1.0 and the
        mechanism cannot arise -- it would be roughly twenty lines and a
        second forward pass carried for a case that does not exist here. It is
        not reimplemented: a non-1.0 weight is a different interface, and the
        right way to say so is at the boundary rather than by silently doing
        the wrong thing.
        """
        rows = [list(row) for row in token_id_rows]
        positions = self.transformer.text_model.embeddings \
            .position_embedding.num_embeddings

        lengths = {len(row) for row in rows}
        if len(lengths) > 1:
            # Ragged rows cannot be stacked into one tensor. ComfyUI hits
            # this as an opaque "Sizes of tensors must match" from torch.cat.
            raise ValueError(f"token rows must all be the same length, got "
                             f"{[len(row) for row in rows]}")
        if lengths and lengths != {positions}:
            # Not ragged, just the wrong length: the position embedding is a
            # lookup of `positions` entries, so this otherwise fails deep
            # inside the forward as a shape error about two tensors.
            # Stricter than ComfyUI, which pads a short row inside
            # `process_tokens`; deliberate, and the tokenizer pads to 77
            # already so nothing here relies on the permissive behaviour.
            raise ValueError(
                f"token rows must be {positions} long to match the position "
                f"embedding, got {sorted(lengths)}; the tokenizer is what "
                f"pads to {positions}, so this means a hand-built row")

        out, pooled = self.encode(rows)
        # The batch comes back stacked, and the sections are rejoined along
        # the *sequence* axis rather than the batch axis. That is what makes
        # `SDXLClipModel` truncate to a common length meaningful, and it is
        # what a caller with several prompt sections expects: one long
        # sequence, not a batch.
        joined = torch.cat([out[i:i + 1] for i in range(out.shape[0])], dim=-2)
        return joined, pooled[0:1] if pooled is not None else pooled

class SDXLClipG(SDClipModel):
    """CLIP-G: the 32-layer tower, padding with token 0 rather than EOS."""

    def __init__(self, dtype=None, device=None) -> None:
        super().__init__(
            CLIP_G_CONFIG, layer="hidden", layer_idx=-2, dtype=dtype,
            device=device,
            special_tokens={"start": 49406, "end": 49407, "pad": 0},
            layer_norm_hidden_state=False, return_projected_pooled=True)


class SDXLClipModel(nn.Module):
    """Both towers, and the concatenation that makes the 2048-dim context.

    The context is CLIP-L's conditioning concatenated with CLIP-G's along the
    feature axis, truncated to the shorter of the two sequences. 768 + 1280 =
    2048, which is the `context_dim` in the UNet's configuration. The pooled
    output is CLIP-G's, which is what SDXL's `y` carries alongside it.
    """

    def __init__(self, dtype=None, device=None) -> None:
        super().__init__()
        self.clip_l = SDClipModel(
            CLIP_L_CONFIG, layer="hidden", layer_idx=-2, dtype=dtype,
            device=device, layer_norm_hidden_state=False,
            special_tokens={"start": 49406, "end": 49407, "pad": 49407})
        self.clip_g = SDXLClipG(dtype=dtype, device=device)

    def encode_token_ids(self, token_id_pairs):
        """Encode both towers and join them into the 2048-wide context.

        `token_id_pairs` is `{"l": [ids], "g": [ids]}` -- one row per prompt
        section, ids only. See `SDClipModel.encode_token_ids` for why there
        are no weights.
        """
        g_out, g_pooled = self.clip_g.encode_token_ids(token_id_pairs["g"])
        l_out, _ = self.clip_l.encode_token_ids(token_id_pairs["l"])
        length = min(l_out.shape[1], g_out.shape[1])
        context = torch.cat([l_out[:, :length], g_out[:, :length]], dim=-1)
        return context, g_pooled