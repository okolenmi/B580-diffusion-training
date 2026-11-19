"""Baking saved adapters into a loaded denoiser, for every node that loads one for something other than training them."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from families.api.parts import AdapterMerge


def apply_merges(family: Any, denoiser: Any, merges: Sequence[AdapterMerge], models: Callable[[], Any], task: Any = None) -> None:
    """Merges each adapter in order; ``models`` fetches the model-library service when the first file is needed, ``task`` opens a named
    stretch of work (``context.task``)."""
    for merge in merges:
        scheme = family.schemes.get(merge.format)
        if scheme is None or not hasattr(scheme, "key"):
            raise ValueError(f"Merge Adapter: {family.id} cannot merge the {merge.format!r} format; choose from kohya")
        path = models().resolve("loras", merge.file)
        if task is not None:
            with task(f"Merging {merge.file}"):
                report = _merge(family, denoiser, scheme, path, merge)
        else:
            report = _merge(family, denoiser, scheme, path, merge)
        if report.merged == 0:
            raise ValueError(f"Merge Adapter: nothing in {merge.file} matches this network (wrong family or key format?)")


def _merge(family: Any, denoiser: Any, scheme: Any, path: Any, merge: AdapterMerge):
    state, _ = family.read_adapter(str(path))
    return family.merge_adapter(denoiser.module(), scheme, state, merge.strength)
