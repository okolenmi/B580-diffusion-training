"""PrimitiveCacheThrashDetector: notice at runtime that the oneDNN primitive
cache is too small for the shapes this run actually uses.

Why this exists. `nodes/xpu_env.py` sizes
`ONEDNN_PRIMITIVE_CACHE_CAPACITY` from a measured constant
(`PRIMITIVES_PER_SHAPE_MEASURED`, 29-35 primitives per shape) that came from
one configuration. Rank, DoRA, target-module selection and optimizer strategy
all change how many primitives a step creates, so the number that is right for
the configuration it was measured on is only a guess for any other -- and a
guess that is too low fails in a way nothing else reports.

What "too low" looks like was measured on the B580 (`shapes diversity
problem/MEASURED-shape-stall.md`): with capacity 1024 and 44 shapes, a step
whose shape had been seen before took 3.790 s against 0.873 s for a step that
repeated the previous shape -- a 4.34x revisit penalty. Raising the capacity
removed it (4.34x -> 0.99x) while leaving *first* sightings untouched
(3.99 s -> 3.85 s), which is what identified the cost as primitive-cache
eviction rather than compilation. First sightings staying slow is the
signature: they are JIT, and no cache size fixes them.

So the detector compares exactly those two populations and stays quiet when
only first sightings are slow, because that is a different problem with a
different answer (fewer shapes, i.e. shape bucketing) and warning about the
cache there would send someone to the wrong knob.

Deliberately one warning per run, not a rate. A condition that is true for the
rest of the run would otherwise print on every step; the ratio does not change
after the first few revisits, because the cache is either big enough or it is
not.
"""

from __future__ import annotations

import statistics
from typing import Optional

#: The knob this detector is about, named in the warning so the reader does
#: not have to know which of the project's env vars it is.
CAPACITY_ENV_VAR = "ONEDNN_PRIMITIVE_CACHE_CAPACITY"

#: A revisit this many times slower than a repeat is worth a warning. 1.5 is
#: deliberately loose: step time is noisy, and a detector that fires on a 20%
#: wobble teaches people to ignore it.
DEFAULT_REVISIT_RATIO = 1.5

#: Revisits needed before deciding. Below three, one slow step that happened
#: to follow a shape change sets the median, and the "median" of two numbers
#: is not evidence of anything.
DEFAULT_MIN_REVISITS = 3

#: Repeats needed to have a baseline at all. Without one there is no ratio,
#: and a detector that invented a denominator would fire on a dataset that
#: never repeats a shape -- where "revisit" means something else entirely.
DEFAULT_MIN_REPEATS = 1


class PrimitiveCacheThrashDetector:
    """Feed it (shape, seconds) per step; get back a warning string, once.

    `shape` is the latent shape as a string ("96x64"), or None when the step
    did not report one -- which is ignored rather than guessed at, because a
    step with an unknown shape cannot be classified and misclassifying it
    would put a wrong number into a median.
    """

    def __init__(self, revisit_ratio: float = DEFAULT_REVISIT_RATIO,
                 min_revisits: int = DEFAULT_MIN_REVISITS,
                 min_repeats: int = DEFAULT_MIN_REPEATS):
        if revisit_ratio <= 1.0:
            raise ValueError(
                f"PrimitiveCacheThrashDetector: revisit_ratio="
                f"{revisit_ratio} must be > 1, or every run that repeats a "
                f"shape would warn")
        if min_revisits < 1 or min_repeats < 1:
            raise ValueError(
                f"PrimitiveCacheThrashDetector: min_revisits={min_revisits} "
                f"min_repeats={min_repeats}, both must be >= 1")
        self._revisit_ratio = float(revisit_ratio)
        self._min_revisits = int(min_revisits)
        self._min_repeats = int(min_repeats)
        self._seen: set = set()
        self._prev: object = None
        self._repeat: list[float] = []
        self._revisit: list[float] = []
        self._first: list[float] = []
        self._warned = False
        self._warning: Optional[str] = None

    def observe(self, shape, seconds: Optional[float]) -> Optional[str]:
        """Record one step. Returns the warning string the first time thrash
        is detected and None every other time, including after.

        A caller that wants it as a log record rather than a print can log the
        returned string; returning it (rather than printing here) is also why
        this class does not import logging or print -- it is a measurement,
        and the decision about where measurements go belongs to the caller.
        """
        if self._warned or shape is None or seconds is None:
            return None
        try:
            dt = float(seconds)
        except (TypeError, ValueError):
            return None
        # A non-positive or non-finite step time is not a slow step, it is a
        # broken measurement. Including it would let one bad clock reading
        # drag a median and manufacture a detection.
        if not (dt > 0.0) or dt != dt or dt in (float("inf"), float("-inf")):
            return None

        if self._prev is not None:
            if shape == self._prev:
                self._repeat.append(dt)
            elif shape in self._seen:
                self._revisit.append(dt)
            else:
                self._first.append(dt)
        # The first shape a run sees has no predecessor, so it is a first
        # sighting by definition and is recorded as such.
        if self._prev is None:
            self._first.append(dt)
        self._seen.add(shape)
        self._prev = shape

        warning = self._maybe_warn()
        if warning is not None:
            self._warned = True
            self._warning = warning
        return warning

    def _maybe_warn(self) -> Optional[str]:
        if len(self._revisit) < self._min_revisits:
            return None
        if len(self._repeat) < self._min_repeats:
            return None
        repeat_median = statistics.median(self._repeat)
        revisit_median = statistics.median(self._revisit)
        if repeat_median <= 0.0:
            return None
        ratio = revisit_median / repeat_median
        if ratio <= self._revisit_ratio:
            return None
        return (
            f"WARNING: revisits of an already-compiled shape are "
            f"{ratio:.2f}x slower than repeats ({revisit_median:.3f} s vs "
            f"{repeat_median:.3f} s, medians of {len(self._revisit)} and "
            f"{len(self._repeat)} steps). That is the signature of the "
            f"oneDNN primitive cache evicting and recompiling a shape it has "
            f"already compiled -- first sightings stay slow for a different "
            f"reason (JIT) and no cache size touches those. Raise "
            f"{CAPACITY_ENV_VAR} (nodes/xpu_env.py sizes it from a constant "
            f"measured on one configuration; rank, DoRA, targets and "
            f"optimizer strategy all change how many primitives a step "
            f"needs). Reported once per run.")

    @property
    def warned(self) -> bool:
        return self._warned

    @property
    def warning(self) -> Optional[str]:
        return self._warning

    def report(self) -> dict:
        """Counts and medians, for a caller that wants the numbers whether or
        not the warning fired. Always present, so "it did not warn" is
        distinguishable from "it was not looking"."""
        def med(values):
            return statistics.median(values) if values else None
        return {
            "distinct_shapes": len(self._seen),
            "first": len(self._first),
            "repeat": len(self._repeat),
            "revisit": len(self._revisit),
            "repeat_median_sec": med(self._repeat),
            "revisit_median_sec": med(self._revisit),
            "first_median_sec": med(self._first),
            "revisit_ratio": (med(self._revisit) / med(self._repeat)
                              if self._repeat and self._revisit
                              and med(self._repeat) > 0.0 else None),
            "warned": self._warned,
        }


def format_warning(detector: PrimitiveCacheThrashDetector) -> Optional[str]:
    """The detector's warning, prefixed with the trainer that produced it.

    A prefix rather than a bare line, because these prints interleave with the
    loader's and the residency controller's and an unprefixed warning in that
    stream is genuinely hard to trace back.
    """
    warning = detector.warning
    if warning is None:
        return None
    return f"  [PrimitiveCacheThrash] {warning}"
