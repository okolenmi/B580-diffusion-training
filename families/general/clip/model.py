"""The CLIP text tower, written once and configured for L and G.

Parameter names follow the Hugging Face layout (``text_model.embeddings.*``,
``text_model.encoder.layers.N.*``, ``text_model.final_layer_norm``, ``text_projection``), so
``checkpoint.py`` only has to translate other spellings into it.

Choices SDXL depends on, none of which raise an error when wrong:
* the conditioning is the penultimate layer's hidden state, and the final norm is not applied to it;
* the pooled vector is the final-normed hidden state at the first EOS position, then projected;
* CLIP-L uses quick-GELU and CLIP-G plain GELU;
* attention is causal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class CLIPConfig:
    hidden: int
    intermediate: int
    heads: int
    layers: int
    activation: str
    positions: int = 77
    vocab: int = 49408
    eos_id: int = 49407
    projection: bool = True


CLIP_L = CLIPConfig(hidden=768, intermediate=3072, heads=12, layers=12, activation="quick_gelu", projection=False)
CLIP_G = CLIPConfig(hidden=1280, intermediate=5120, heads=20, layers=32, activation="gelu")

ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),
    "gelu": F.gelu,
    "gelu_tanh": lambda a: F.gelu(a, approximate="tanh"),
}


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        q, k, v = (p(x).view(b, n, self.heads, d // self.heads).transpose(1, 2)
                   for p in (self.q_proj, self.k_proj, self.v_proj))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).reshape(b, n, d))


class MLP(nn.Module):
    def __init__(self, dim: int, inner: int, activation: str) -> None:
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, inner), nn.Linear(inner, dim)
        self.activation = ACTIVATIONS[activation]

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class Layer(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.layer_norm1, self.layer_norm2 = nn.LayerNorm(config.hidden), nn.LayerNorm(config.hidden)
        self.self_attn = Attention(config.hidden, config.heads)
        self.mlp = MLP(config.hidden, config.intermediate, config.activation)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        return x + self.mlp(self.layer_norm2(x))


class Embeddings(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab, config.hidden)
        self.position_embedding = nn.Embedding(config.positions, config.hidden)


class Encoder(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(Layer(config) for _ in range(config.layers))


class TextModel(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.embeddings, self.encoder = Embeddings(config), Encoder(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden)


class CLIPTextTower(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.text_model = TextModel(config)
        if config.projection:
            self.text_projection = nn.Linear(config.hidden, config.hidden, bias=False)

    def forward(self, tokens: Tensor, *, layer: int = -2, norm_hidden: bool = False) -> tuple[Tensor, Tensor | None]:
        """(hidden state at ``layer``, projected pooled vector or None).

        ``tokens`` is (B, positions) of ids. ``layer`` counts from the end when negative.
        """
        cfg, tm = self.config, self.text_model
        weight_dtype = tm.final_layer_norm.weight.dtype
        x = (tm.embeddings.token_embedding(tokens) + tm.embeddings.position_embedding.weight).to(weight_dtype)
        tap = layer if layer >= 0 else cfg.layers + layer
        if not 0 <= tap < cfg.layers:
            raise ValueError(f"layer {layer} is out of range for {cfg.layers} layers")
        hidden = None
        for index, block in enumerate(tm.encoder.layers):
            x = block(x)
            if index == tap:
                hidden = x
        final = tm.final_layer_norm(x)
        if norm_hidden:
            hidden = tm.final_layer_norm(hidden)
        pooled = None
        if cfg.projection:
            eos = (tokens == cfg.eos_id).int().argmax(dim=-1)
            pooled = self.text_projection(final[torch.arange(final.shape[0], device=final.device), eos])
        return hidden, pooled
