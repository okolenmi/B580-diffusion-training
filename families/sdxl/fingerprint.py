"""How to recognise SDXL files from a header: a full checkpoint or any single part of one."""

from __future__ import annotations

from ..api.detection import Match, Rejected
from ..api.header import HeaderIndex
from ..api.parts import PartInfo, PartKind
from ..general.fingerprint import clip_text, first_prefix, indexed_count, vae_autoencoder

FAMILY_ID = "sdxl"
LABEL = "SDXL"

_UNET_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "")
_SDXL_ADM = 2816
_SDXL_CONTEXT = 2048
_SDXL_Z = 4


def _info(index: HeaderIndex, kind: PartKind, role: str, prefix: str, config: dict, exclude=()) -> PartInfo:
    keys = [k for k in index.keys_with(prefix) if not any(k.startswith(prefix + e) for e in exclude)]
    nbytes = sum(index.entries[k].nbytes for k in keys)
    return PartInfo(kind, role, prefix, len(keys), index.dominant_dtype(prefix), nbytes, config)


def _unet(index: HeaderIndex) -> PartInfo | None:
    prefix = first_prefix(index, _UNET_PREFIXES, "input_blocks.0.0.weight")
    if prefix is None:
        return None
    first = index.shape(f"{prefix}input_blocks.0.0.weight")
    label = index.shape(f"{prefix}label_emb.0.0.weight")
    context = index.shape(f"{prefix}input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight")
    if label is None or label[1] != _SDXL_ADM or not context or context[1] != _SDXL_CONTEXT:
        width = (index.shape(f"{prefix}input_blocks.1.1.transformer_blocks.0.attn2.to_k.weight") or (0, 0))[1]
        why = f"UNet cross-attention width is {width} (SD 1.5 style)" if width == 768 else \
            "UNet does not have SDXL's size conditioning and 2048-wide cross-attention"
        raise Rejected(why)
    config = {"model_channels": first[0] if first else None, "adm_in_channels": label[1],
              "context_dim": context[1], "input_blocks": indexed_count(index, f"{prefix}input_blocks.")}
    return _info(index, PartKind.DENOISER, "denoiser", prefix, config)


def _clips(index: HeaderIndex) -> list[PartInfo]:
    out: list[PartInfo] = []
    for base, role, hidden in (("conditioner.embedders.0.", "clip_l", 768), ("conditioner.embedders.1.model.", "clip_g", 1280)):
        found = clip_text(index, base)
        if found and found.config["hidden"] == hidden:
            out.append(_info(index, PartKind.TEXT_ENCODER, role, base, {**found.config, "layout": found.layout}))
    if not out:
        # single-file encoders: CLIP-L / CLIP-G saved on their own, bare or under a transformer prefix
        for base in ("", "transformer.", "clip_l.", "clip_g."):
            found = clip_text(index, base)
            if found and found.config["hidden"] in (768, 1280):
                role = "clip_l" if found.config["hidden"] == 768 else "clip_g"
                out.append(_info(index, PartKind.TEXT_ENCODER, role, found.prefix, {**found.config, "layout": found.layout}))
                break
    return out


def _vae(index: HeaderIndex) -> PartInfo | None:
    for base in ("first_stage_model.", ""):
        found = vae_autoencoder(index, base)
        if found and found.config.get("z_channels") == _SDXL_Z:
            return _info(index, PartKind.VAE, "vae", base, {**found.config, "layout": found.layout,
                                                           "scale": 0.13025})
    return None


def match(index: HeaderIndex) -> Match | None:
    parts: list[PartInfo] = []
    unet = _unet(index)
    if unet:
        parts.append(unet)
    parts.extend(_clips(index))
    vae = _vae(index)
    if vae:
        parts.append(vae)
    if not parts:
        return None
    roles = {p.role for p in parts}
    if {"denoiser", "clip_l", "clip_g", "vae"} <= roles:
        score = 1.0
    elif "denoiser" in roles:
        score = 0.95
    elif roles & {"clip_l", "clip_g"}:
        score = 0.8
    else:
        score = 0.6
    return Match(FAMILY_ID, score, tuple(parts), labels={"denoiser": "UNet"}, family_label=LABEL)
