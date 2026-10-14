"""The only module that names the families. Families are imported lazily, one fingerprint at a time."""

from __future__ import annotations

import importlib
from typing import Any

from .api.detection import Detection, Matcher, detect
from .api.header import ModelFileError
from .api.library import ModelLibrary

FAMILY_IDS = ("sdxl", "anima")


def _module_name(family_id: str, leaf: str) -> str:
    root = ".".join((__package__ or "").split(".")[:-1])  # package that holds `families`
    return ".".join(part for part in (root, "families", family_id, leaf) if part)


def fingerprint(family_id: str) -> Any:
    if family_id not in FAMILY_IDS:
        raise KeyError(f"unknown family {family_id!r}")
    return importlib.import_module(_module_name(family_id, "fingerprint"))


def matchers() -> dict[str, Matcher]:
    return {fid: fingerprint(fid).match for fid in FAMILY_IDS}


def list_families() -> list[dict]:
    return [{"id": fid, "label": fingerprint(fid).LABEL} for fid in FAMILY_IDS]


def detect_file(library: ModelLibrary, scope: str, relative: str, family: str = "auto") -> Detection:
    """Detect the contents of one scope file. File problems become an error Detection, never a raise."""
    try:
        index = library.header(scope, relative)
    except ModelFileError as exc:
        return Detection("error", None, message=str(exc), error_kind=exc.kind)
    return detect(index, matchers(), family)
