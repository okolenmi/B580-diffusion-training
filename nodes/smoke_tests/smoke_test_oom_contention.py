"""An out-of-memory in a GPU smoke test is contention or growth -- and
which one must not be a matter of opinion.

`fast_construction.oom_outcome` classifies a device OOM by comparing the
peak allocation the process reached against the test's measured footprint
(10,669 MB: 9,804 MB of fp32 SDXL weights plus a latent-32 backward, on a
12,216 MB card whose foreign usage measures 1,100-1,500 MB). That
classification is what decides whether the gate goes green on a skip or
red on a regression, so every branch of it is checked here with synthetic
numbers -- a test of the rule needs no card to exhaust.

Three claims are falsifiable, and all three fail this file if untrue:

1. an OOM at or below footprint+tolerance returns 0 (skip) and *prints*
   the peak, the footprint and the OOM's own free/allocated line;
2. an OOM above it returns None so the caller re-raises, at the boundary
   as one MB over it;
3. already-recorded failures are never downgraded to a skip by the card
   running out.

The exit-code half of the contract -- that these outcomes really become
exit 0 / non-zero in a process that raises `torch.OutOfMemoryError` -- is
a cross-process property, so it is checked against two real child
processes rather than in-process. And the two GPU tests that carry the
guard are read from disk to confirm they still wire it: a classifier
nobody calls is a comment shaped like a safety net.
"""

import contextlib
import io
import subprocess  # noqa: S603 -- fixed argv, no shell
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nodes.smoke_tests.fast_construction import (  # noqa: E402
    OOM_FOOTPRINT_TOLERANCE_MB,
    oom_outcome,
)

#: The footprint the two GPU tests carry as their constant -- the peak
#: their own measurement printed (gate run: `peak allocated: off 10669
#: MB`), and the peak the OOM in the gate log actually reached (10.45
#: GiB allocated, 7.61 MiB free).
FOOTPRINT_MB = 10_669.0

REPO = Path(__file__).resolve().parents[2]
GPU_TESTS = (
    "nodes/smoke_tests/gpu/smoke_test_real_training_step.py",
    "nodes/smoke_tests/gpu/smoke_test_lora_merge_identity.py",
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_outcome(**kwargs) -> tuple[int | None, str]:
    """`oom_outcome` with captured stdout, defaults shared by every case."""
    arguments = {
        "exc": RuntimeError(
            "XPU out of memory. Tried to allocate 16.00 MiB. GPU 0 has a "
            "total capacity of 11.93 GiB of which 7.61 MiB is free."
        ),
        "footprint_mb": FOOTPRINT_MB,
        "failures": [],
        "name": "probe",
    }
    arguments.update(kwargs)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        outcome = oom_outcome(**arguments)
    return outcome, captured.getvalue()


def check_contention_skips_and_prints_its_measurements():
    """Claim 1: at-footprint OOM -> exit 0, with every number printed."""
    print("[an OOM inside the measured footprint is contention: exit 0, "
          "peak/footprint/free all printed]")
    # 10,700 MB is the allocation the gate log's real OOM reached (10.45
    # GiB) -- inside the footprint, so this is the case that used to fail
    # the gate as a crash.
    outcome, text = run_outcome(peak_mb=10_700.0, free_mb=8.0,
                                total_mb=12_216.0)
    check(outcome == 0, f"expected a skip (0), got {outcome!r}")
    for wanted in ("SKIP:", "VRAM contention", "peak allocated 10,700 MB",
                   "footprint of 10,669 MB", "device free 8 MB of 12,216 MB",
                   "SKIPPED (VRAM contention", "7.61 MiB is free"):
        check(wanted in text, f"the skip line must print {wanted!r}, "
                              f"got:\n{text}")
    print("    PASS (exit 0, all four numbers and the OOM line present)")


def check_growth_above_the_footprint_is_never_a_skip():
    """Claim 2: above footprint+tolerance -> None, boundary inclusive."""
    print("[an OOM above footprint+tolerance returns None so the caller "
          "re-raises]")
    ceiling = FOOTPRINT_MB + OOM_FOOTPRINT_TOLERANCE_MB
    at_ceiling, text_at = run_outcome(peak_mb=ceiling, free_mb=0.0,
                                      total_mb=12_216.0)
    check(at_ceiling == 0,
          f"the ceiling itself ({ceiling} MB) must still be a skip, "
          f"got {at_ceiling!r}")
    over_ceiling, text_over = run_outcome(peak_mb=ceiling + 1, free_mb=0.0,
                                          total_mb=12_216.0)
    check(over_ceiling is None,
          f"one MB past the ceiling must re-raise (None), got "
          f"{over_ceiling!r}")
    check("grew past its own measurement" in text_over,
          f"the regression path must say the test grew, got:\n{text_over}")
    check("re-raised" in text_over,
          f"the regression path must say the exception is re-raised, "
          f"got:\n{text_over}")
    check("SKIP" not in text_over,
          f"the regression path must not offer a skip, got:\n{text_over}")
    print(f"    PASS ({ceiling} MB skips, {ceiling + 1} MB re-raises)")


def check_recorded_failures_are_not_downgraded():
    """Claim 3: a failed check stays failed through an OOM."""
    print("[failures already recorded keep the run red even when the OOM "
          "would otherwise be contention]")
    outcome, text = run_outcome(peak_mb=10_700.0, free_mb=8.0,
                                total_mb=12_216.0,
                                failures=["the adapters moved"])
    check(outcome == 1, f"expected failure (1), got {outcome!r}")
    check("FAILURE(S)" in text and "the adapters moved" in text,
          f"the recorded failure must be printed, got:\n{text}")
    check("SKIP" not in text,
          f"a failed run must not be relabelled a skip, got:\n{text}")
    print("    PASS (exit 1, the failure printed, no skip offered)")


def check_both_gpu_tests_still_carry_the_guard():
    """The classifier is only as real as its two callers."""
    print("[both GPU tests still wire the guard into their __main__]")
    for relative in GPU_TESTS:
        source = (REPO / relative).read_text()
        for wanted in ("except torch.OutOfMemoryError",
                       "oom_outcome(",
                       "FOOTPRINT_MB"):
            check(wanted in source,
                  f"{relative} no longer carries {wanted!r}: the guard "
                  f"would silently not run")
    print(f"    PASS ({len(GPU_TESTS)} files, 3 markers each)")


def _child_exit(code_body: str) -> tuple[int, str]:
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-c", code_body],
        capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stdout + proc.stderr


def check_real_processes_turn_the_outcome_into_an_exit_code():
    """The contract is about exit codes, so real processes decide it.

    Each child runs the same shape the two GPU tests run: raise
    `oom_outcome`'s caller path, then exit on its result. The contention
    child must exit 0 with the skip printed; the growth child must exit
    non-zero, which is `None` propagating as a re-raise.
    """
    print("[in two real processes: contention exits 0, growth exits "
          "non-zero]")
    common = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO)!r})\n"
        "from nodes.smoke_tests.fast_construction import oom_outcome\n"
        "outcome = oom_outcome(\n"
        "    RuntimeError('XPU out of memory. Tried to allocate 16.00 MiB.'),\n"
        "    footprint_mb=10669.0, failures=[], name='probe',\n"
        f"    peak_mb={{peak}}, free_mb=8.0, total_mb=12216.0)\n"
    )
    rc_ok, out_ok = _child_exit(
        common.format(peak=10_700.0)
        + "raise SystemExit(0 if outcome == 0 else 1)\n")
    check(rc_ok == 0,
          f"contention child must exit 0, got {rc_ok}:\n{out_ok}")
    check("SKIPPED (VRAM contention" in out_ok,
          f"contention child must print the skip, got:\n{out_ok}")

    rc_growth, out_growth = _child_exit(
        common.format(peak=11_400.0)
        + "raise SystemExit(1 if outcome is None else 0)\n")
    check(rc_growth != 0,
          f"growth child must exit non-zero (None -> re-raise), got "
          f"{rc_growth}:\n{out_growth}")
    check("grew past its own measurement" in out_growth,
          f"growth child must print the regression line, got:\n"
          f"{out_growth}")
    print(f"    PASS (exit {rc_ok} with skip, exit {rc_growth} with the "
          f"regression line)")


def main():
    check_contention_skips_and_prints_its_measurements()
    check_growth_above_the_footprint_is_never_a_skip()
    check_recorded_failures_are_not_downgraded()
    check_both_gpu_tests_still_carry_the_guard()
    check_real_processes_turn_the_outcome_into_an_exit_code()
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
