"""Timesteps per image that start high and fall to the normal count."""

from __future__ import annotations

import math


class RegionLadder:
    """The first batches of a run use more timesteps per image than the rest. The count starts at ``start`` and falls in equal
    parts to the noiser's own count over ``span`` batches. ``skip`` is how far into the ladder the run already is: batches
    already trained when training is continued, so a continued run does not climb the ladder again."""

    def __init__(self, start: int, span: int = 120, skip: int = 0) -> None:
        if start < 1 or span < 1 or skip < 0:
            raise ValueError("a ladder needs a start of at least 1, a span of at least 1 batch and a skip of 0 or more")
        self.start, self.span, self.skip = start, span, skip

    def at(self, batch: int, final: int) -> int:
        """Timesteps per image of the ``batch``-th batch of the run when the run ends at ``final``."""
        left = self.span - (batch + self.skip)
        if self.start <= final or left <= 0:
            return final
        return final + math.ceil((self.start - final) * left / self.span)

    def peak(self, final: int) -> int:
        """The most timesteps per image any batch of the run has."""
        return max(final, self.start) if self.skip < self.span else final
