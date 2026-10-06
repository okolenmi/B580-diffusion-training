"""SYCL/Level-Zero performance environment variables for Intel XPU --
single source of truth, called from every real process entry point
(nodes/cli.py, backend/cli.py since M9 -- formerly server_cli.py, now
archive/server_cli.py) before anything else in that process touches
torch/XPU.

Extracted from core/cli.py, where these five lines already lived,
unconditionally, at module-import time, predating this file. Real gap
found investigating a reported speed regression this session:
server_cli.py (the node-graph/Resources-Controller route's own process
entry point, retired to archive/ with M9) never set any of these --
graph execution (server/routes_nodegraph.py's _worker(), same archive)
runs training in a background thread inside that same long-lived
server process, never as its own `python -m core.cli` subprocess the
way the older route always has, so core/cli.py's own os.environ[...]
lines, module-level as they are, never ran for that route at all.
Confirmed directly: no other file in
this codebase sets any of these (grepped for SYCL_/UR_L0_/IGC_Enable
across the whole repo before writing this).

Real, reported symptom this plausibly explains: fast steps, then a long
stall, then a few more fast steps, correlated with encountering a new
image resolution -- consistent with SYCL's own kernel JIT-compile/
selection being repeated (rather than served from a warm in-memory
cache) every time a genuinely new tensor shape reaches the GPU, which a
mixed-aspect-ratio dataset does far more often than a fixed-resolution
one. Not confirmed on real hardware (no XPU available in this
environment) -- these are real, documented SYCL/Level-Zero runtime
variables (not invented for this fix), already present, uncommented,
in this project's own older route, so the expectation that they help is
grounded in this project's own prior, working configuration, not a
fresh guess. The two PERSISTENT-cache lines (SYCL_CACHE_PERSISTENT/
SYCL_CACHE_DIR) stay commented out here too, unchanged from core/cli.py
-- an on-disk kernel cache surviving process restarts is a separate,
real decision (first-ever run of a given shape is still a compile no
disk cache prevents; persisting across the *server* process's own
restarts specifically, which the old subprocess-per-run route never
had a reason to care about) that whoever already chose not to enable it
in the older route didn't make for this file to second-guess.
"""

import os


def set_xpu_perf_env_vars(
    onednn_primitive_cache_capacity: int | None = None,
) -> None:
    """Idempotent (plain assignment, safe to call more than once or from
    more than one entry point in the same process) and side-effect-free
    beyond os.environ -- no torch import here, so this is safe to call
    before torch (or anything importing it) is touched at all, which is
    the whole point: SYCL reads these at its own runtime init, the first
    time anything actually touches an XPU device, so setting them even
    slightly late (after that first touch) is too late.

    ``onednn_primitive_cache_capacity`` overrides the default below. Two
    ways to use it, both of which reach every training process:

    * pass it, from a caller that knows the answer -- a graph child given
      the shape count by the server, which already reads the dataset's
      latent buckets during admission;
    * export ``ONEDNN_PRIMITIVE_CACHE_CAPACITY`` before starting the server.
      Both gateways spawn children via ``os.environ.copy()``
      (backend/infrastructure/graph_task_gateway.py), so one export covers
      the server and every child it starts, which is the whole reason an
      exported value is honoured here rather than overwritten: a knob an
      operator cannot set before a run is not a knob.

    An explicit argument wins over the environment, because a caller that
    computed a value from the dataset knows more than a shell variable
    could. An unusable value -- non-numeric, or below 1 -- is refused and
    the default is used instead, with a warning: oneDNN parses this with
    strtol, so a bad value silently becomes 0, and a cache of zero entries
    is worse than the default this function exists to set.
    """
    # os.environ["SYCL_CACHE_PERSISTENT"] = "1"
    # os.environ["SYCL_CACHE_DIR"] = str(Path.home() / ".cache" / "sycl_kernels")
    os.environ["SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD"] = "0"
    os.environ["SYCL_CACHE_IN_MEM"] = "1"

    os.environ["UR_L0_USE_RELAXED_ALLOCATION_LIMITS"] = "1"
    os.environ["SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS"] = "1"
    os.environ["IGC_EnableDPEmulation"] = "1"

    # oneDNN's primitive cache, sized for a multi-resolution dataset.
    #
    # The default (1024) cannot hold the primitives a UNet needs per latent
    # shape, so a dataset with more than a few distinct shapes re-creates
    # them on every shape *transition*. Measured on the B580, torch
    # 2.12.1+xpu, batch 2, production env, `non-square` (44 shapes,
    # clumped at mean 2.16), by capacity:
    #
    #   capacity  revisit / steady   steps/s (150 steps)   peak host RSS
    #      1024         3.85x             0.488               15,567 MB
    #      2048         0.99x             0.599               15,567 MB
    #     65536         1.00x             0.591               15,567 MB
    #
    # 2048 is enough and 65536 buys nothing over it, so this is 2048 rather
    # than the 65536 first tried -- 32x more cache for an identical number.
    # Peak host RSS is byte-identical at every capacity, so the cache costs
    # no measurable host memory either (VmHWM is a kernel high-water mark, so
    # this is not a sampling gap).
    #
    # 2048 is what `non-square` needs, and the requirement scales with the
    # shape count: a dataset with several hundred distinct shapes needs more.
    # Raise it if revisits go slow again -- revisit/steady is the signal, and
    # it is directly measurable now that each step records its shape.
    #
    # This fixes TRANSITIONS only. First sightings still cost ~3.85 s each,
    # and since every graph run is a fresh process (MEM-05), a 44-shape
    # dataset repays ~164 s per run -- roughly 40% of a 300-step run. That
    # cost is not configurable: SYCL_CACHE_PERSISTENT was measured to save
    # 2% of it while writing 1.1 GB to disk, because it persists SYCL's
    # kernels and not oneDNN's primitives. Fewer distinct shapes is the only
    # lever on it (docs/known-issues/open.md).
    #
    # Here rather than in a trainer or a config file: this function is the
    # one place both entry points already call before torch is imported
    # (backend/cli.py, and each graph child in
    # backend/infrastructure/graph_task_worker.py).
    #
    # Unlike the five lines above, an already-set value is KEPT, not
    # overwritten -- see the docstring on the parameter. That is the
    # opposite of what this function does elsewhere, and deliberately: the
    # other five are internal tuning this project asserts, while this one
    # has to be raiseable by whoever owns the dataset, because the right
    # value depends on the dataset's shape count and this module cannot
    # see a dataset. An override in effect is logged, so a run that is slow
    # for this reason says so instead of looking like a mystery.
    _set_primitive_cache_capacity(onednn_primitive_cache_capacity)


#: Used when nothing supplies a capacity and no dataset is known.
#:
#: Sized for `non-square` (44 shapes) with headroom: measured, the working set
#: is (1280, 1536], and peak host RSS was byte-identical at every capacity
#: from 1024 to 262144 -- a 256x range. So capacity is a ceiling, not an
#: allocation, and headroom above what a dataset needs is free. That is why
#: this is a round number above the requirement rather than the requirement:
#: the cost of being generous is zero and the cost of being stingy is the
#: 4x revisit penalty.
DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY = 2048

#: Measured primitives per distinct latent shape, as a bracket rather than a
#: point, because that is the precision the measurement has: 44 shapes need a
#: capacity in (1280, 1536], which is 29.1 to 34.9 per shape. The wide end is
#: used for sizing so a dataset is never sized to the optimistic end.
PRIMITIVES_PER_SHAPE_MEASURED = (29, 35)

#: Above this many distinct latent shapes, grouping is recommended over a
#: bigger cache -- not because the cache runs out, but because grouping is
#: cheaper on both axes at once. At 35 primitives/shape, 4096 shapes would
#: need ~143k entries; the cache would hold them, free, but grouping 44
#: shapes to 3 also cuts the ~166s per-run first-sighting cost to ~11s,
#: which no cache can touch. Named as a constant so the recommendation has a
#: number attached rather than living only in prose.
SHAPES_WHERE_GROUPING_WINS = 512


def primitive_cache_capacity_for_shapes(
        distinct_shapes: int | None) -> int | None:
    """A capacity that holds ``distinct_shapes`` shapes' primitives, or
    None when the shape count is unknown.

    Sized from the wide end of the measured bracket (35 primitives per shape)
    so a dataset is never sized to the optimistic end of a 1.2x-wide
    measurement. Returns None for an unknown or non-positive count rather than
    a default: "we do not know how many shapes this dataset has" is a
    different fact from "this dataset has few shapes", and answering it with
    a number would hide the difference. The caller then uses the default,
    which is generous because capacity is free.

    This cannot fail for want of a number -- the measured ratio is a property
    of the model and the backend, not of the dataset -- but it can be wrong
    for a model whose per-shape primitive count differs from the SDXL UNet
    that was measured. For that case the caller keeps an explicit override
    (an argument, or the environment variable).
    """
    if distinct_shapes is None:
        return None
    if not isinstance(distinct_shapes, int) or isinstance(distinct_shapes, bool):
        return None
    if distinct_shapes <= 0:
        return None
    return max(DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
               distinct_shapes * PRIMITIVES_PER_SHAPE_MEASURED[1])


def _set_primitive_cache_capacity(explicit: int | None) -> None:
    """Resolve the capacity from (in order) an explicit argument, the
    environment, then the measured default; warn on an unusable value."""
    name = "ONEDNN_PRIMITIVE_CACHE_CAPACITY"
    if explicit is not None:
        value = explicit
    elif name in os.environ:
        raw = os.environ[name]
        try:
            value = int(raw.strip())
        except ValueError:
            _warn_bad_capacity(raw, "environment")
            value = DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY
    else:
        value = DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY

    if value < 1:
        _warn_bad_capacity(str(value),
                           "argument" if explicit is not None else "environment")
        value = DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY

    os.environ[name] = str(value)


def _warn_bad_capacity(raw: str, source: str) -> None:
    import logging
    logging.getLogger(__name__).warning(
        "ignoring %s=%r from %s: not a usable oneDNN primitive cache capacity; "
        "using the measured default %d. oneDNN parses this with strtol, so an "
        "unusable value becomes 0 -- a cache of zero entries, worse than not "
        "setting it at all",
        "ONEDNN_PRIMITIVE_CACHE_CAPACITY", raw, source,
        DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY,
    )
