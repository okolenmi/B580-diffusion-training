"""Notice at runtime that the oneDNN primitive cache is too small for the shapes the run uses.

A step whose shape was seen before (a revisit) that is much slower than a step repeating the previous shape means the
cache evicted a compiled shape and had to build it again. When only first sightings are slow that is compilation, which
no cache size fixes (the answer there is fewer shapes), so it stays quiet. One warning per run: the ratio does not change
after the first few revisits.
"""

from __future__ import annotations

import statistics
from collections.abc import Hashable

from ..xpu_env import CACHE_VAR


class PrimitiveCacheMonitor:
    def __init__(self, ratio: float = 1.5, min_revisits: int = 3, min_repeats: int = 1):
        if ratio <= 1.0 or min_revisits < 1 or min_repeats < 1:
            raise ValueError("the ratio must be above 1 (or every run that repeats a shape would warn) and the minimums at least 1")
        self.ratio, self.min_revisits, self.min_repeats = ratio, min_revisits, min_repeats
        self._seen: set[Hashable] = set()
        self._previous: Hashable | None = None
        self._repeat: list[float] = []
        self._revisit: list[float] = []
        self.warned = False

    def observe(self, shape: Hashable, seconds: float) -> str | None:
        """Record one step; the warning text the first time thrash shows, None otherwise."""
        if self.warned or not seconds > 0.0 or seconds == float("inf"):
            return None
        if shape == self._previous:
            self._repeat.append(seconds)
        elif shape in self._seen:
            self._revisit.append(seconds)
        self._seen.add(shape)
        self._previous = shape
        if len(self._revisit) < self.min_revisits or len(self._repeat) < self.min_repeats:
            return None
        repeat, revisit = statistics.median(self._repeat), statistics.median(self._revisit)
        if revisit / repeat <= self.ratio:
            return None
        self.warned = True
        return (f"revisits of an already compiled shape are {revisit / repeat:.2f}x slower than repeats ({revisit:.3f} s vs {repeat:.3f} s, "
                f"medians of {len(self._revisit)} and {len(self._repeat)} steps): the oneDNN primitive cache is evicting shapes it "
                f"already built. Raise {CACHE_VAR} (rank, DoRA, targets and optimizer all change how many primitives a shape needs) "
                f"or lower max_shapes. Reported once.")
