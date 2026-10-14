"""How to recognise Anima files from a header.

Decisive keys come from ComfyUI's model detection as read in rework/audit/anima-model-facts.md:
a Cosmos transformer has ``blocks.0.mlp.layer1.weight``; Anima adds
``llm_adapter.blocks.0.cross_attn.q_proj.weight``. Qwen3 and Wan-style VAE layouts are
recollection and must be confirmed against real files before anything relies on them.
"""

from __future__ import annotations

from ..api.detection import Match, Rejected
from ..api.header import HeaderIndex
from ..api.parts import PartInfo, PartKind
from ..general.fingerprint import first_prefix, indexed_count, qwen3_text, vae_autoencoder

FAMILY_ID = "anima"
LABEL = "Anima"

_DIT_PREFIXES = ("net.", "model.diffusion_model.", "")
_ADAPTER_PROBE = "llm_adapter.blocks.0.cross_attn.q_proj.weight"
_Z = 16


def _info(index: HeaderIndex, kind: PartKind, role: str, prefix: str, config: dict, exclude=()) -> PartInfo:
    keys = [k for k in index.keys_with(prefix) if not any(k.startswith(prefix + e) for e in exclude)]
    return PartInfo(kind, role, prefix, len(keys), index.dominant_dtype(prefix),
                    sum(index.entries[k].nbytes for k in keys), config)


def _dit(index: HeaderIndex) -> list[PartInfo]:
    prefix = first_prefix(index, _DIT_PREFIXES, "blocks.0.mlp.layer1.weight")
    if prefix is None:
        return []
    if f"{prefix}{_ADAPTER_PROBE}" not in index:
        raise Rejected("a Cosmos transformer without the LLM adapter is not Anima")
    embed = index.shape(f"{prefix}x_embedder.proj.1.weight")
    config = {"blocks": indexed_count(index, f"{prefix}blocks."), "width": embed[0] if embed else None,
              "adapter_blocks": indexed_count(index, f"{prefix}llm_adapter.blocks.")}
    return [
        _info(index, PartKind.DENOISER, "denoiser", prefix, config, exclude=("llm_adapter.",)),
        _info(index, PartKind.ADAPTER, "adapter", f"{prefix}llm_adapter.", {"blocks": config["adapter_blocks"]}),
    ]


def match(index: HeaderIndex) -> Match | None:
    parts = _dit(index)
    for base in ("model.", ""):
        found = qwen3_text(index, base)
        if found:
            parts.append(_info(index, PartKind.TEXT_ENCODER, "qwen3", base, {**found.config, "layout": found.layout}))
            break
    for base in ("", "vae."):
        found = vae_autoencoder(index, base)
        if found and found.config.get("z_channels") == _Z:
            parts.append(_info(index, PartKind.VAE, "vae", base, {**found.config, "layout": found.layout}))
            break
    if not parts:
        return None
    roles = {p.role for p in parts}
    score = 1.0 if "denoiser" in roles else 0.7
    return Match(FAMILY_ID, score, tuple(parts), labels={"denoiser": "DiT"}, family_label=LABEL)
