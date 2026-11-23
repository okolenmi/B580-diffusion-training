"""The run's shelf of kept denoisers, shared by the generation nodes."""

from __future__ import annotations

from typing import Any

from engine.generate.shelf import DenoiserShelf


def shelf_of(context: Any, device: str) -> DenoiserShelf:
    """The shelf of this run, made on first use; a run without a runtime gets one that keeps nothing."""
    runtime = context.runtime
    residency = getattr(runtime, "residency", None)
    if runtime is None or residency is None:
        return DenoiserShelf(None, device)
    if runtime.denoisers is None:
        runtime.denoisers = DenoiserShelf(residency, device)
    return runtime.denoisers
