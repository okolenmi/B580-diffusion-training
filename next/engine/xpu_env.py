"""SYCL and oneDNN environment variables for Intel GPUs. Call before the first step runs; no torch import needed.

The measurements behind the values are in rework/71-memory.md.
"""

from __future__ import annotations

import logging
import os

CACHE_VAR = "ONEDNN_PRIMITIVE_CACHE_CAPACITY"
DEFAULT_CACHE_CAPACITY = 2048
PRIMITIVES_PER_SHAPE = 35  # upper end of the measured 29-35 range, so a dataset is never sized to the optimistic end
SHAPES_WHERE_GROUPING_WINS = 512

_log = logging.getLogger(__name__)
_explicit = False  # the capacity came from the environment rather than from this module


def set_perf_env(capacity: int | None = None) -> None:
    """Idempotent. An exported ``ONEDNN_PRIMITIVE_CACHE_CAPACITY`` is kept; an explicit ``capacity`` wins over it."""
    os.environ["SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD"] = "0"
    os.environ["SYCL_CACHE_IN_MEM"] = "1"
    os.environ["UR_L0_USE_RELAXED_ALLOCATION_LIMITS"] = "1"
    os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
    os.environ["IGC_EnableDPEmulation"] = "1"
    _set_capacity(capacity)


def capacity_for_shapes(distinct_shapes: int | None) -> int | None:
    """Capacity that holds the primitives of ``distinct_shapes`` latent shapes; None when the count is unknown."""
    if not isinstance(distinct_shapes, int) or isinstance(distinct_shapes, bool) or distinct_shapes <= 0:
        return None
    return max(DEFAULT_CACHE_CAPACITY, distinct_shapes * PRIMITIVES_PER_SHAPE)


def size_cache_for_shapes(distinct_shapes: int | None) -> int | None:
    """Size the primitive cache once the dataset is known; the capacity in effect, or None when nothing applied.

    Over-counting shapes is safe (capacity is a ceiling) and under-counting is slow, so callers may pass the
    count of stored shapes. An exported value is respected, with a warning when it is too small.
    """
    required = capacity_for_shapes(distinct_shapes)
    if required is None:
        return None
    current = os.environ.get(CACHE_VAR)
    if current is not None and _explicit:
        try:
            have = int(current.strip())
        except ValueError:
            return None
        if have < required:
            _log.warning("%s=%s was set explicitly but %d distinct latent shapes need about %d entries; steps that are slow "
                         "when a shape comes back are what a too-small cache looks like", CACHE_VAR, current, distinct_shapes, required)
        return have
    _set_capacity(required)
    return required


def _set_capacity(explicit: int | None) -> None:
    global _explicit
    if explicit is not None:
        value = explicit
    elif CACHE_VAR in os.environ:
        _explicit = True
        try:
            value = int(os.environ[CACHE_VAR].strip())
        except ValueError:
            _warn_unusable(os.environ[CACHE_VAR])
            value = DEFAULT_CACHE_CAPACITY
    else:
        value = DEFAULT_CACHE_CAPACITY
    if value < 1:
        _warn_unusable(str(value))
        value = DEFAULT_CACHE_CAPACITY
    os.environ[CACHE_VAR] = str(value)


def _warn_unusable(raw: str) -> None:
    _log.warning("ignoring %s=%r: oneDNN reads it with strtol, so an unusable value becomes a cache of zero entries; "
                 "using %d", CACHE_VAR, raw, DEFAULT_CACHE_CAPACITY)
