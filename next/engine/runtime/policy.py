"""What a run is allowed to take from the card: the ``memory.*`` and ``runtime.*`` settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...graph.settings import AUTO


class MemoryBudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class MemoryPolicy:
    """The ``memory.*`` and ``runtime.*`` settings. ``limit_mb`` None means the whole card.

    ``margin_mb`` is the one safety margin: memory kept free below the limit for what cannot be predicted (kernels
    and primitives of a shape not seen yet, driver growth). Every threshold below is the limit minus measured
    outside memory minus this, and nothing adds a second margin on top.
    """

    limit_mb: float | None = None
    floor_mb: float = 0.0
    strict: bool = True
    margin_mb: float = 512.0
    calibration_steps: int = 3
    empty_cache_every: int = 1

    @classmethod
    def from_settings(cls, settings: Any) -> MemoryPolicy:
        limit = settings.get("memory.vram_max_mb")
        return cls(None if limit in (None, AUTO) else float(limit), float(settings.get("memory.vram_min_mb") or 0),
                   bool(settings.get("memory.strict")), float(settings.get("memory.margin_mb")),
                   int(settings.get("runtime.residency.calibration_steps")), int(settings.get("runtime.empty_cache_every_n_steps")))
