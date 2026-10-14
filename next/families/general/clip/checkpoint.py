"""Translating the key spellings CLIP checkpoints use into ``CLIPTextTower``'s names."""

from __future__ import annotations

from typing import MutableMapping

from torch import Tensor

LAYOUTS = ("hf", "hf_nested", "openclip")

_TOP = {
    "positional_embedding": "text_model.embeddings.position_embedding.weight",
    "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
    "ln_final.weight": "text_model.final_layer_norm.weight",
    "ln_final.bias": "text_model.final_layer_norm.bias",
}
_BLOCK = {
    "ln_1": "layer_norm1", "ln_2": "layer_norm2", "mlp.c_fc": "mlp.fc1", "mlp.c_proj": "mlp.fc2",
    "attn.out_proj": "self_attn.out_proj",
}
_QKV = ("q_proj", "k_proj", "v_proj")


def to_tower_state(tensors: MutableMapping[str, Tensor], layout: str, layers: int) -> dict[str, Tensor]:
    """Rename ``tensors`` (keys relative to the part's prefix) for ``CLIPTextTower``. The input is not modified.

    * ``hf``: already in the tower's names.
    * ``hf_nested``: the tower sits under ``transformer.``, which is dropped.
    * ``openclip``: resblocks, fused attention input projection (split in three) and a projection
      stored as a matrix that multiplies on the right (transposed into a Linear weight).
    """
    if layout not in LAYOUTS:
        raise ValueError(f"unknown CLIP layout {layout!r}; choose from {LAYOUTS}")
    src = dict(tensors)
    if layout == "hf":
        return src
    if layout == "hf_nested":
        return {(k[len("transformer."):] if k.startswith("transformer.") else k): v for k, v in src.items()}
    out: dict[str, Tensor] = {}
    for old, new in _TOP.items():
        if old in src:
            out[new] = src.pop(old)
    for i in range(layers):
        base, dest = f"transformer.resblocks.{i}.", f"text_model.encoder.layers.{i}."
        for old, new in _BLOCK.items():
            for suffix in ("weight", "bias"):
                if f"{base}{old}.{suffix}" in src:
                    out[f"{dest}{new}.{suffix}"] = src.pop(f"{base}{old}.{suffix}")
        for suffix in ("weight", "bias"):
            packed = src.pop(f"{base}attn.in_proj_{suffix}", None)
            if packed is not None:
                third = packed.shape[0] // 3
                for j, name in enumerate(_QKV):
                    out[f"{dest}self_attn.{name}.{suffix}"] = packed[third * j:third * (j + 1)].clone()
    if "text_projection" in src:
        out["text_projection.weight"] = src.pop("text_projection").transpose(0, 1).contiguous()
    elif "text_projection.weight" in src:
        out["text_projection.weight"] = src.pop("text_projection.weight")
    return out
