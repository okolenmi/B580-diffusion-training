"""Uniform asset listing/browsing/upload for the graph editor's file
pickers and the LoRA "Save As" dialog.

Every relative path accepted here goes through paths.py's sandboxed
resolvers (resolve_safe_model_path / resolve_safe_dataset_path) -- this
module never builds a path itself and never uses the permissive
resolve_model_path/resolve_dataset_path. The graph editor is explicitly
meant to be reachable from another device on the network, so every path
string it hands over is treated as untrusted input, the same posture a
file-upload handler on any public web app would take -- not just here to
be tidy.

Datasets are read-only from this module's point of view: a managed
dataset is a structured multi-file directory built by the Dataset
Manager / data pipeline, not something you'd browse into, create
subfolders in, or drop a raw upload into.
"""

from __future__ import annotations

from pathlib import Path

import paths

KINDS = ("checkpoint", "lora", "dataset")
BROWSABLE_KINDS = ("checkpoint", "lora")  # kinds that support browse/mkdir/upload


def _base_dir(kind: str) -> Path:
    if kind == "checkpoint":
        return paths.get_checkpoints_dir()
    if kind == "lora":
        return paths.get_loras_dir()
    if kind == "dataset":
        return paths.get_datasets_dir()
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def _safe_resolve(kind: str, relative_str: str) -> Path:
    if kind == "dataset":
        return paths.resolve_safe_dataset_path(relative_str)
    return paths.resolve_safe_model_path(relative_str, kind)


def list_options(kind: str) -> dict:
    """Flat list for a <select> picker -- existing files (checkpoint/lora
    list includes subfolder-nested ones already, via list_model_files'
    recursive glob; dataset list is the library's top-level dataset names)."""
    base_dir = _base_dir(kind)

    if kind in ("checkpoint", "lora"):
        options = [{"value": f, "label": f} for f in paths.list_model_files(kind)]
        return {"kind": kind, "base_dir": str(base_dir), "options": options,
                "upload_supported": True, "browse_supported": True}

    from manager.dataset import ManagedDatasetLibrary
    library = ManagedDatasetLibrary(base_dir)
    options = [{"value": d["name"], "label": d["name"] + (f" -- {d['description']}" if d["description"] else "")}
               for d in library.list_datasets()]
    return {"kind": kind, "base_dir": str(base_dir), "options": options,
            "upload_supported": False, "browse_supported": False}


def browse(kind: str, relative_dir: str = "") -> dict:
    """Immediate children of relative_dir (folders and .safetensors files)
    -- what the "Save As" dialog walks through. Root is relative_dir=""."""
    if kind not in BROWSABLE_KINDS:
        raise ValueError(f"Browsing is only supported for {BROWSABLE_KINDS}, not {kind!r}")
    base = _base_dir(kind).resolve()
    target = _safe_resolve(kind, relative_dir) if relative_dir else base
    if not target.exists():
        return {"kind": kind, "path": relative_dir, "folders": [], "files": []}
    if not target.is_dir():
        raise ValueError(f"Not a directory: {relative_dir!r}")

    folders, files = [], []
    for child in sorted(target.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            if child.name == "resume" and target == base:
                continue  # auto-managed working files, same exclusion as list_model_files
            folders.append(child.name)
        elif child.suffix == ".safetensors":
            files.append(child.name)
    return {"kind": kind, "path": relative_dir, "folders": folders, "files": files}


def make_subfolder(kind: str, relative_path: str) -> str:
    if kind not in BROWSABLE_KINDS:
        raise ValueError(f"Creating folders is only supported for {BROWSABLE_KINDS}, not {kind!r}")
    resolved = _safe_resolve(kind, relative_path)
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved)


def save_upload(kind: str, relative_path: str, content: bytes) -> str:
    if kind not in BROWSABLE_KINDS:
        raise ValueError(f"Upload is only supported for {BROWSABLE_KINDS}, not {kind!r}")
    resolved = _safe_resolve(kind, relative_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_bytes(content)
    return str(resolved)


def inspect(kind: str, relative_path: str) -> dict:
    """Per-component (unet/clip/vae) dtype for a checkpoint, read
    cheaply from the file's own header -- see
    nodes/model/resource_inspection.py's own docstring for the real
    mechanics (safetensors' header-only read, no tensor data touched)
    and why this is cheap even for a multi-GB file. Powers the
    Resources Controller's per-input dtype readouts
    (docs/resources_controller_redesign_plan.md, Phase 2) without
    loading the resource itself.

    Only 'checkpoint' is supported today. A LoRA file's own meaningful
    fields (its own dtype, its rank) are real, separate, not-yet-built
    follow-up work -- a LoRA file's key format
    (lora_unet_X.lora_down.weight) isn't the raw SDXL prefix format
    checkpoint inspection is grounded in, so it needs its own
    inspection function, not a silent expansion of this one. Same
    reasoning as this project's own preference throughout for building
    the one real thing before generalizing past it -- see
    resource_inspection.py's module docstring.

    Response is deliberately narrow: exactly the detected fields per
    component, nothing else -- no raw header dump, no full key list.
    This is what "protected" means for this endpoint specifically: what
    comes back is a fixed, small contract, not a passthrough of
    whatever the file's header happens to contain. Same sandboxing as
    every other function in this module (_safe_resolve) -- this field
    is reachable from the graph editor over the network, same posture
    as browse()/save_upload() above.
    """
    if kind != "checkpoint":
        raise ValueError(
            f"Resource inspection currently only supports kind='checkpoint', not {kind!r} -- "
            f"LoRA/dataset inspection is real, separate, not-yet-built follow-up work."
        )
    resolved = _safe_resolve(kind, relative_path)
    if not resolved.exists():
        raise ValueError(f"No such file: {relative_path!r}")
    if not resolved.is_file():
        raise ValueError(f"Not a file: {relative_path!r}")

    from nodes.model.resource_inspection import dtype_to_str, inspect_checkpoint_dtypes

    try:
        per_component = inspect_checkpoint_dtypes(resolved)
    except Exception as e:
        raise ValueError(f"{relative_path!r} doesn't look like a valid safetensors file ({e}).") from e

    return {
        "kind": kind,
        "path": relative_path,
        "components": {
            name: {"dtype": dtype_to_str(info.dtype), "key_count": info.key_count}
            for name, info in per_component.items()
        },
    }
