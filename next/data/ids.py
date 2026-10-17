"""Stable identities: images by content, latent spaces by the VAE that makes them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..families.api.header import HeaderIndex

CHUNK = 1 << 20
SAMPLE_SEP = "#"


def hash_file(path: str | Path) -> str:
    """Sixteen hex digits of the file's sha256. Moving or renaming an image keeps its id."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def sample_id(image_id: str, crop: int = 0) -> str:
    """Crop 0 is the bare image id; further crops of a long image get a ``#k`` suffix."""
    return image_id if crop == 0 else f"{image_id}{SAMPLE_SEP}{crop}"


def image_of(sample: str) -> str:
    return sample.split(SAMPLE_SEP, 1)[0]


def spec_fields(spec: Any) -> dict:
    """The part of a latent spec that decides whether two caches are interchangeable."""
    return {"channels": int(spec.channels), "spatial_factor": int(spec.spatial_factor),
            "scale": float(spec.scale), "shift": float(getattr(spec, "shift", 0.0))}


def vae_identity(header: HeaderIndex, prefix: str, spec: Any, probes: int = 3) -> str:
    """Hash of a VAE's layout under ``prefix``, a few of its weights, and its latent spec.

    The layout alone cannot tell a fine-tuned VAE from the original (same keys, same shapes), so
    ``probes`` tensors spread over the sorted key list are hashed too. The file is read only for
    those tensors.
    """
    return weights_identity(header, prefix, spec_fields(spec), probes)


def weights_identity(header: HeaderIndex, prefix: str, spec: Any, probes: int = 3) -> str:
    """Hash of the layout under ``prefix``, a few of its weights, and ``spec`` (anything JSON-able that changes the
    meaning of what is computed from them)."""
    import torch
    from safetensors import safe_open

    keys = sorted(k for k in header.entries if k.startswith(prefix))
    if not keys:
        raise ValueError(f"no tensors under prefix {prefix!r} in this file")
    layout = [(k[len(prefix):], list(header.entries[k].shape), header.entries[k].dtype) for k in keys]
    digest = hashlib.sha256(json.dumps({"layout": layout, "spec": spec}, sort_keys=True).encode())
    picks = sorted({keys[i * (len(keys) - 1) // max(probes - 1, 1)] for i in range(probes)})
    with safe_open(header.path, framework="pt") as handle:
        for key in picks:
            digest.update(handle.get_tensor(key).contiguous().view(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()[:16]
