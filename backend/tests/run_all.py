"""Run every backend test file, in parallel, each in a fresh interpreter.

    python backend/tests/run_all.py
    python backend/tests/run_all.py --jobs 1     # the old serial behaviour
    python backend/tests/run_all.py --jobs 4 --only test_supervisor.py

**Why parallel.** The suite was 21.5s serial on this machine, with the
five slowest files accounting for 53% of it, while the CPU sat at 58%
idle -- the tests are short and independent, and the cost was paying for
them one after another. Expected floor is the slowest single file
(`test_api_graphs.py`, 2.77s) plus pool overhead.

Two properties had to survive, because `run_all.py` is what the gate
calls:

* **Each file gets its own `TMPDIR`, and it is removed afterwards.** The
  test files create scratch with `tempfile.mkdtemp`, which returns a name
  and hands back no handle, so nothing in any of them cleans up after
  itself. Pointing `TMPDIR` at a directory this script owns contains that
  litter -- and 26 files run at once is 26 litters per run rather than one
  shared mess. It is also what makes the files independent, which is what
  makes parallelising them safe: without it, two files could pick the same
  path.
* **The exit code**, and the two summary lines the gate greps for.

**Output order is preserved.** Results are collected in file order, not
completion order, because a log that interleaves 26 files is unreadable
and makes a failure much harder to find -- which is the opposite of what
running them faster is for.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_one(job: tuple[str, str]) -> tuple[str, int, str]:
    """Run one test file in its own TMPDIR. Never raises."""
    name, scratch_root = job
    scratch = Path(scratch_root) / Path(name).stem
    scratch.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "TMPDIR": str(scratch)}
    try:
        result = subprocess.run(
            [sys.executable, str(HERE / name)],
            capture_output=True, text=True, env=env, timeout=600,
        )
        return name, result.returncode, (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return name, 124, f"TIMED OUT after 600s: {name}"
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1),
        help="parallel workers; 1 runs the files in sequence",
    )
    parser.add_argument(
        "--only", nargs="*", help="run just these file(s), by name",
    )
    args = parser.parse_args()

    test_files = sorted(HERE.glob("test_*.py"))
    if args.only:
        wanted = set(args.only)
        test_files = [p for p in test_files if p.name in wanted]
        missing = wanted - {p.name for p in test_files}
        if missing:
            print(f"no such test file(s): {', '.join(sorted(missing))}")
            return 1
    if not test_files:
        print("no test files found")
        return 1

    scratch_root = tempfile.mkdtemp(prefix="backend-testrun-")
    jobs = [(p.name, scratch_root) for p in test_files]
    failed: list[str] = []
    try:
        if args.jobs <= 1:
            results: list[tuple[str, int, str]] = []
            for job in jobs:
                results.append(run_one(job))
                name, rc, output = results[-1]
                print(f"\n=== {name} ===")
                print(output.rstrip())
                if rc:
                    failed.append(name)
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                # map() yields in submission order, which is the point:
                # see the module docstring on output ordering.
                for name, rc, output in pool.map(run_one, jobs, chunksize=1):
                    print(f"\n=== {name} ===")
                    print(output.rstrip())
                    if rc:
                        failed.append(name)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    print("\n" + "=" * 60)
    if failed:
        print(f"BACKEND TESTS: {len(failed)}/{len(test_files)} FILE(S) FAILED")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"BACKEND TESTS: ALL {len(test_files)} FILE(S) PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())