"""Loading a denoiser from a checkpoint file for the instruments that work on a real model (no dataset, no graph)."""

from __future__ import annotations

from pathlib import Path

import torch

from ..engine.xpu_env import set_perf_env
from ..families.api.library import ModelLibrary
from ..families.api.parts import make_part
from ..families.registry import detect_file, get_family


def load_parts(models: str, file: str, scope: str = "checkpoints"):
    """The family and the model parts (by role) that ``file`` (relative to ``models/scope``, or to ``models`` when it has no such subfolder) holds."""
    set_perf_env()
    root = Path(models) / scope
    library = ModelLibrary({scope: [root]} if root.is_dir() else {scope: [models]})
    found = detect_file(library, scope, file)
    if not found.ok:
        raise SystemExit(f"cannot use {file}: {found.message}")
    path = str(library.resolve(scope, file))
    return get_family(found.family_id), {p.role: make_part(found.family_id, path, p) for p in found.parts}


def load_denoiser(models: str, file: str, device: str, dtype=torch.bfloat16, scope: str = "checkpoints"):
    """The family's denoiser wrapper for ``file``."""
    family, parts = load_parts(models, file, scope)
    return family.load_denoiser(parts["denoiser"], device=device, dtype=dtype)


def checkpoint_path(models: str, file: str, scope: str = "checkpoints") -> str:
    """The path of ``file`` the way ``load_denoiser`` resolves it."""
    root = Path(models) / scope
    return str(ModelLibrary({scope: [root]} if root.is_dir() else {scope: [models]}).resolve(scope, file))
