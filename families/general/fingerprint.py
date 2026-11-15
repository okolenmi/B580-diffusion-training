"""Recognising parts that several families share, from a header (no torch).

Each function inspects one candidate location and returns a small description or None. A family's
own fingerprint decides which of these parts it claims. Facts about key layouts come from the
formats the old loader already read (ldm / OpenCLIP / Hugging Face naming); the Qwen3 and
Wan-style VAE layouts are recollection, to be confirmed against real files (audit/anima-model-facts.md).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..api.header import HeaderIndex


@dataclass(frozen=True)
class Found:
    prefix: str
    layout: str
    config: dict


def first_prefix(index: HeaderIndex, candidates: tuple[str, ...], probe: str) -> str | None:
    """The first candidate prefix under which ``probe`` exists (longest candidates first is the caller's job)."""
    for prefix in candidates:
        if f"{prefix}{probe}" in index:
            return prefix
    return None


def indexed_count(index: HeaderIndex, prefix: str) -> int:
    """How many distinct numeric segments follow ``prefix`` ('blocks.' -> blocks.0, blocks.1, ...)."""
    seen: set[int] = set()
    for key in index.keys_with(prefix):
        head = key[len(prefix):].split(".", 1)[0]
        if head.isdigit():
            seen.add(int(head))
    return len(seen)


def clip_text(index: HeaderIndex, base: str) -> Found | None:
    """A CLIP text tower under ``base``: Hugging Face, nested-transformer or OpenCLIP naming."""
    layouts = (
        ("hf", f"{base}text_model.embeddings.token_embedding.weight", f"{base}text_model.encoder.layers."),
        ("hf_nested", f"{base}transformer.text_model.embeddings.token_embedding.weight",
         f"{base}transformer.text_model.encoder.layers."),
        ("openclip", f"{base}token_embedding.weight", f"{base}transformer.resblocks."),
    )
    for name, probe, layers in layouts:
        shape = index.shape(probe)
        if shape and len(shape) == 2:
            return Found(base, name, {"hidden": shape[1], "vocab": shape[0], "layers": indexed_count(index, layers)})
    return None


def qwen3_text(index: HeaderIndex, base: str) -> Found | None:
    """A Qwen3 language model (Hugging Face naming). The q_norm/k_norm keys tell Qwen3 from Qwen2."""
    embed = index.shape(f"{base}embed_tokens.weight")
    if not embed or len(embed) != 2 or f"{base}layers.0.self_attn.q_norm.weight" not in index:
        return None
    return Found(base, "hf", {"hidden": embed[1], "vocab": embed[0],
                              "layers": indexed_count(index, f"{base}layers.")})


def vae_autoencoder(index: HeaderIndex, base: str) -> Found | None:
    """An autoencoder under ``base``. config['z_channels'] is the latent channel count."""
    quant = index.shape(f"{base}quant_conv.weight")
    if f"{base}encoder.down.0.block.0.conv1.weight" in index and quant:
        return Found(base, "ldm", {"z_channels": quant[0] // 2})
    if f"{base}encoder.down_blocks.0.resnets.0.conv1.weight" in index and quant:
        return Found(base, "diffusers", {"z_channels": quant[0] // 2})
    if f"{base}encoder.conv1.weight" in index and any(
            k.startswith(f"{base}decoder.upsamples.") for k in index.keys_with(f"{base}decoder.")):
        conv2 = index.shape(f"{base}conv2.weight") or index.shape(f"{base}post_quant_conv.weight")
        return Found(base, "wan", {"z_channels": conv2[0] if conv2 else None})
    return None
