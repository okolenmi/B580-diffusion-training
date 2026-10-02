"""Run every smoke_test_*.py in this directory and print a combined summary.

    python nodes/smoke_tests/run_all.py                     # parallel
    python nodes/smoke_tests/run_all.py --jobs 1           # strictly sequential
    python nodes/smoke_tests/run_all.py memory adafactor    # filename-substring filters
    python nodes/smoke_tests/run_all.py --serial-only       # just the sensitive list

Exits 0 only if every test that ran exited 0.

Convenience only -- each file remains independently runnable and
independently meaningful; this doesn't replace reading a given test's own
output when something fails, it just saves typing all the filenames.

`run_tests.py` calls this with filter arguments and depends on both the
filter semantics and the exit code, so those are load-bearing.

Why parallel, and why not all of it
-----------------------------------

Measured serially: **119.9s over 66 files**, with the cost spread evenly
-- the ten slowest account for 26% and the longest single file is 4.4s.
That shape parallelises well, and with nothing longer than 4.4s there is
no long-tail problem: the floor is the slowest file plus pool overhead.

The reason it is not *all* parallel is the list below. The maintainer of
this runner is right that these tests do not collectively strain 12 GB of
VRAM -- measured, they are small. The residual risk is not exhaustion, it
is **interference**: several of them assert on device state
(`xpu_empty_cache`/`xpu_synchronize` ordering, residency transitions,
offload behaviour). Two of those running at once are measuring each
other's allocations, so a failure would mean nothing and a pass would
prove less than it appears to. They cost ~15s in total, which is a cheap
price for not having to wonder which was which.

So: an explicit, auditable serial list, and everything else in parallel.

Multiprocessing uses the **spawn** context deliberately. The default
`fork` start method crashes the pool here -- a forked child inheriting a
torch/XPU-initialised parent is not safe, and it failed as
`ConnectionResetError` from the forkserver rather than anything legible.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent

#: Tests that run one at a time, with the reason for each. Kept explicit
#: rather than pattern-matched so that adding a test to this list is a
#: decision someone can see, and so a test that only *looks* like these
#: does not get serialised by accident.
SERIAL_TESTS: dict[str, str] = {
    "smoke_test_device_context_equivalence.py":
        "asserts the xpu_empty_cache/xpu_synchronize call order",
    "smoke_test_memory_manager.py":
        "asserts on allocator and residency state",
    "smoke_test_sdxl_text_encoder_offload.py":
        "asserts an offload actually released device memory",
    "smoke_test_composed_adafactor.py":
        "heavy XPU allocation; longest test in the suite at ~4s",
    "smoke_test_composed_came.py":
        "heavy XPU allocation; ~4.4s, the slowest in the suite",
    "smoke_test_composed_adamw.py":
        "heavy XPU allocation",
    "smoke_test_adafactor_tiny_parameter_gap.py":
        "measures a parameter gap on device; sensitive to neighbours",
    "smoke_test_nf4_lora_layer.py":
        "quantised weights held on device",
}


def discover_tests(filters: list[str]) -> list[Path]:
    tests = sorted(p for p in _HERE.glob("smoke_test_*.py"))
    if filters:
        tests = [p for p in tests if any(f in p.name for f in filters)]
    return tests


def run_one(job: tuple[str, str]) -> tuple[str, int, str]:
    """Run one smoke test in its own TMPDIR. Never raises.

    The per-file TMPDIR is the same reasoning as the backend runner's: the
    tests create scratch with `tempfile.mkdtemp`, which hands back no
    handle, so nothing cleans up after itself. Containing that in a
    directory this process owns is also what makes the files independent
    enough to run at once -- without it two of them could pick the same
    path.
    """
    name, scratch_root = job
    scratch = Path(scratch_root) / Path(name).stem
    scratch.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": str(scratch)}
    try:
        proc = subprocess.run(
            [sys.executable, str(_HERE / name)],
            capture_output=True, text=True, env=env, timeout=900,
        )
        return name, proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return name, 124, f"TIMED OUT after 900s: {name}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filters", nargs="*",
                        help="filename substrings; anything matching runs")
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1),
                        help="parallel workers for the non-sensitive tests")
    parser.add_argument("--serial-only", action="store_true",
                        help="run only the sensitive list, one at a time")
    parser.add_argument("--quiet", action="store_true",
                        help="summary only; per-test output is suppressed")
    args = parser.parse_args()

    tests = discover_tests(args.filters)
    if not tests:
        print(f"No smoke_test_*.py files matched filters {args.filters!r} in {_HERE}")
        sys.exit(1)

    serial = [t for t in tests if t.name in SERIAL_TESTS]
    unknown = [t.name for t in tests if t.name in SERIAL_TESTS
               and not ( _HERE / t.name ).exists()]
    if unknown:  # defensive: a filter can name a file that does not exist
        print(f"SERIAL_TESTS names missing files: {', '.join(unknown)}")
        sys.exit(1)
    parallel = [t for t in tests if t.name not in SERIAL_TESTS]

    if args.serial_only:
        parallel = []

    print(f"Running {len(tests)} test file(s): "
          f"{len(parallel)} parallel (jobs={args.jobs}), "
          f"{len(serial)} one at a time")
    for t in tests:
        tag = "serial " if t.name in SERIAL_TESTS else "parallel"
        print(f"  [{tag}] {t.name}")

    scratch_root = tempfile.mkdtemp(prefix="smoke-testrun-")
    results: list[tuple[str, int]] = []
    try:
        # The sensitive list first and alone, so nothing else is holding
        # device memory while it runs.
        for t in serial:
            name, rc, output = run_one((t.name, scratch_root))
            results.append((name, rc))
            if not args.quiet:
                print(f"\n{'='*70}\n{name}\n  serial: {SERIAL_TESTS[name]}\n{'='*70}")
                print(output.rstrip())

        if parallel:
            jobs = [(t.name, scratch_root) for t in parallel]
            # spawn, not fork: see the module docstring.
            with ProcessPoolExecutor(
                max_workers=max(1, args.jobs),
                mp_context=mp.get_context("spawn"),
            ) as pool:
                # map() yields in submission order, so the log reads in
                # filename order rather than completion order.
                for name, rc, output in pool.map(run_one, jobs, chunksize=1):
                    results.append((name, rc))
                    if not args.quiet:
                        print(f"\n{'='*70}\n{name}\n{'='*70}")
                        print(output.rstrip())
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    failed = [name for name, rc in results if rc != 0]
    for name, rc in results:
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        marker = " [serial]" if name in SERIAL_TESTS else ""
        print(f"  {status}: {name}{marker}")

    if failed:
        print(f"\n{len(failed)}/{len(results)} test file(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} test file(s) passed.")


if __name__ == "__main__":
    main()