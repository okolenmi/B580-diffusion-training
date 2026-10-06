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


def set_xpu_perf_env_vars() -> None:
    """Idempotent (plain assignment, safe to call more than once or from
    more than one entry point in the same process) and side-effect-free
    beyond os.environ -- no torch import here, so this is safe to call
    before torch (or anything importing it) is touched at all, which is
    the whole point: SYCL reads these at its own runtime init, the first
    time anything actually touches an XPU device, so setting them even
    slightly late (after that first touch) is too late."""
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
    # backend/infrastructure/graph_task_worker.py). Unconditional, like the
    # five lines above: a conditional assignment would let a stale exported
    # value silently restore the penalty this line removes.
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "2048"
