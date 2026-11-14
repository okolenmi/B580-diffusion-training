"""Checks nodes/xpu_env.py's set_xpu_perf_env_vars().

There was no test for this function at all, which is why a stale comment in it
could claim the cache variables were "not confirmed on real hardware" after
one of them had been measured on the card (see the ONEDNN comment below, and
docs/known-issues/open.md's non-shape throughput entry). A perf setting with
no test is a perf setting that can be deleted by anyone who does not know what
it was for.

The properties worth pinning are not "the value is 65536" -- a future
measurement may legitimately lower it, and this file must not then fail and
block that. They are:

  * the capacity is set at all, and set to a number oneDNN will accept;
  * it is set to a value that is NOT oneDNN's 1024 default, because 1024 is
    the measured-bad value and silently reverting to it is the failure this
    line exists to prevent;
  * the function stays idempotent and stays torch-free, since it is called
    before torch is imported (backend/cli.py, and each graph child in
    backend/infrastructure/graph_task_worker.py).

Deliberately NOT checked here: that every variable the function sets has an
adjacent comment. The five SYCL variables predate this file and are explained
in the module docstring rather than beside each line, so such a check would
fail on correct existing code and pressure the project into comment churn to
satisfy an assertion with nothing behind it.

Nothing here can check throughput -- that is a hardware measurement, recorded in
the comment and in docs/known-issues/open.md. This checks the setting is
actually applied where the callers will see it.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes import xpu_env
from nodes.xpu_env import set_xpu_perf_env_vars


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def check_onednn_primitive_cache_capacity_is_set():
    print("[ONEDNN_PRIMITIVE_CACHE_CAPACITY is set by set_xpu_perf_env_vars()]")
    os.environ.pop("ONEDNN_PRIMITIVE_CACHE_CAPACITY", None)
    set_xpu_perf_env_vars()
    value = os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY")
    check(value is not None, "not set at all -- the shape-stall fix is gone")
    # oneDNN parses this with strtol; a non-numeric value is silently 0, which
    # would mean a cache of zero entries and a worse stall than the default.
    check(value.strip().isdigit() and int(value) > 0,
          f"must be a positive integer oneDNN can parse, got {value!r}")
    print(f"    value: {value}")
    print("    PASS")


def check_capacity_is_not_the_measured_bad_default():
    print("[the capacity is not oneDNN's 1024 default -- 1024 is the value "
          "measured to cost 4.34x on revisits]")
    set_xpu_perf_env_vars()
    value = int(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"])
    check(value != 1024,
          "ONEDNN_PRIMITIVE_CACHE_CAPACITY is back at oneDNN's 1024 default, "
          "which is the measured-bad value (revisit 4.34x steady on the B580)")
    print("    PASS")


def check_capacity_is_at_least_the_measured_sufficient_value():
    print("[the capacity is at least 2048 -- the smallest value MEASURED to hold "
          "`non-square`'s primitives (1024 measured 3.85x on revisits)]")
    value = int(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"])
    # Not a performance assertion, and deliberately loose: 2048 is what a
    # 44-shape dataset was measured to need, and the requirement scales with
    # the shape count, so a dataset with hundreds of shapes needs more and
    # raising this must not be blocked by this test. What it refuses is a
    # silent drop to the measured-bad default or below.
    check(value >= 2048,
          f"capacity {value} is below the smallest value measured to work "
          f"(2048); 1024 measured revisit 3.85x steady on the B580")
    print(f"    value: {value}")
    print("    PASS")


def check_an_exported_value_is_honoured():
    print("[a value already in the environment is KEPT -- a knob an operator "
          "cannot set before a run is not a knob, and both gateways spawn "
          "children via os.environ.copy()]")
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "99999"
    set_xpu_perf_env_vars()
    check(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] == "99999",
          f"an exported value was overwritten: "
          f"{os.environ['ONEDNN_PRIMITIVE_CACHE_CAPACITY']}")
    print("    PASS")


def check_an_explicit_argument_beats_the_environment():
    print("[an explicit argument wins over the environment -- a caller that "
          "computed the capacity from the dataset knows more than a shell "
          "variable can]")
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "99999"
    set_xpu_perf_env_vars(onednn_primitive_cache_capacity=4096)
    check(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] == "4096",
          os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"])
    print("    PASS")


def check_an_unusable_value_is_refused_and_says_so():
    print("[a non-numeric or below-1 value is refused with a warning -- oneDNN "
          "parses this with strtol, so 'junk' becomes 0 and a cache of zero "
          "entries is worse than not setting it]")
    import io
    import logging
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("nodes.xpu_env")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.WARNING)
    try:
        for bad in ("junk", "0", "-5", ""):
            os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = bad
            set_xpu_perf_env_vars()
            got = os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"]
            check(got == str(xpu_env.DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY),
                  f"{bad!r} should have been refused, got {got!r}")
        check("ignoring" in stream.getvalue(),
              "refusing a value must say so, or a mistyped export looks exactly "
              "like the fix not working")
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)
    print("    PASS")


def check_a_refused_value_never_survives_in_the_environment():
    print("[a refused value does not stay in os.environ -- oneDNN reads the "
          "environment directly, so leaving 'junk' there would be what makes it "
          "parse as 0, not the warning]")
    os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = "junk"
    set_xpu_perf_env_vars()
    check(os.environ["ONEDNN_PRIMITIVE_CACHE_CAPACITY"]
          == str(xpu_env.DEFAULT_ONEDNN_PRIMITIVE_CACHE_CAPACITY),
          f"the refused value must be replaced in os.environ, got "
          f"{os.environ['ONEDNN_PRIMITIVE_CACHE_CAPACITY']!r}")
    print("    PASS")


def check_calling_twice_changes_nothing():
    print("[idempotent: calling twice leaves the same value (both entry points "
          "call it, and a graph child is spawned from a process that called it)]")
    set_xpu_perf_env_vars()
    first = dict(os.environ)
    set_xpu_perf_env_vars()
    check(dict(os.environ) == first, "a second call changed os.environ")
    print("    PASS")


def check_no_torch_is_imported():
    print("[the function imports no torch -- it runs before anything touches an "
          "XPU device, which is the only reason setting these here works]")
    # In a fresh interpreter: import the module and call it, then assert torch
    # is not in sys.modules. Doing it here would be contaminated by the
    # importing test process, so this is a subprocess.
    import subprocess
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from nodes.xpu_env import set_xpu_perf_env_vars;"
        "set_xpu_perf_env_vars();"
        "print('torch' in sys.modules)" % str(Path(__file__).resolve().parents[2])
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, timeout=300).stdout.strip()
    check(out.endswith("False"),
          f"set_xpu_perf_env_vars() pulled torch into sys.modules: {out!r}")
    print("    PASS")


def main():
    check_onednn_primitive_cache_capacity_is_set()
    check_capacity_is_not_the_measured_bad_default()
    check_capacity_is_at_least_the_measured_sufficient_value()
    check_an_exported_value_is_honoured()
    check_an_explicit_argument_beats_the_environment()
    check_an_unusable_value_is_refused_and_says_so()
    check_a_refused_value_never_survives_in_the_environment()
    check_calling_twice_changes_nothing()
    check_no_torch_is_imported()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
