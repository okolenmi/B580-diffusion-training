"""SYCL/Level-Zero performance environment variables for Intel XPU --
single source of truth, called from every real process entry point
(core/cli.py, server_cli.py) before anything else in that process
touches torch/XPU.

Extracted from core/cli.py, where these five lines already lived,
unconditionally, at module-import time, predating this file. Real gap
found investigating a reported speed regression this session:
server_cli.py (the node-graph/Resources-Controller route's own process
entry point) never set any of these -- graph execution
(server/routes_nodegraph.py's _worker()) runs training in a background
thread inside that same long-lived server process, never as its own
`python -m core.cli` subprocess the way the older route always has, so
core/cli.py's own os.environ[...] lines, module-level as they are,
never ran for that route at all. Confirmed directly: no other file in
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
