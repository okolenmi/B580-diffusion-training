#!/usr/bin/env python3
"""Branch coverage for `backend/application` and `backend/infrastructure`.

    python scripts/coverage_report.py

**Report only: it cannot fail the build, and there is no threshold.** The
point is a number that moves and is visible, not a number that gates.
Setting a target today would mean picking one by assertion rather than by
evidence, and a threshold nobody believes is a threshold that gets
commented out.

Two things it does that `coverage run backend/tests/run_all.py` does not:

* **Runs each test file in its own process with its own `TMPDIR`**, the
  same way `run_all.py` does. One combined run would import every test
  module into one interpreter, and these are scripts with module-level
  work, shared module-level singletons (`paths.py`'s resolver, the
  settings store) and no fixtures to isolate them. Measured: they do not
  survive being combined, so the per-process structure is the honest one.
* **Attributes coverage to the file that produced it.** `mutation_report.py`
  consumes this to decide which tests can kill a mutation in a given module,
  rather than a hand-written guess that goes stale.

Output is two tables (statements / branches) for the two directories named
in the round-2 review, plus the overall figure and the worst-covered
modules, because a single percentage hides the one file that needs work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTED = ("backend/application", "backend/infrastructure")


def run_suite() -> None:
    """Coverage for every test file, combined into `.coverage`."""
    tests = sorted((ROOT / "backend" / "tests").glob("test_*.py"))
    if not tests:
        raise SystemExit("no backend test files found")
    for stale in ROOT.glob(".coverage.*"):
        stale.unlink()

    scratch_root = Path(tempfile.mkdtemp(prefix="coverage-"))
    failures: list[str] = []
    try:
        for test in tests:
            scratch = scratch_root / test.stem
            scratch.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "TMPDIR": str(scratch)}
            result = subprocess.run(
                [sys.executable, "-m", "coverage", "run", "--branch",
                 "--parallel-mode", "--source=backend", str(test)],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            if result.returncode != 0:
                failures.append(test.name)
        combine = subprocess.run(
            [sys.executable, "-m", "coverage", "combine"],
            cwd=ROOT, capture_output=True, text=True,
        )
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    if combine.returncode != 0:
        raise SystemExit(f"coverage combine failed: {combine.stderr.strip()}")
    if failures:
        # Coverage still produced data; say which files did not contribute
        # rather than pretending the numbers are complete.
        print(f"coverage_report: WARNING -- {len(failures)} test file(s) "
              f"failed under coverage: {', '.join(failures)}")
        print("The figures below exclude them.")


def report() -> int:
    for directory in REPORTED:
        # `--include` rather than a positional path: passing a directory
        # to `coverage report` makes it try to *read* that directory as a
        # source file and prints "No source for code: Is a directory". The
        # exit code is 0 either way, so the table was simply absent.
        result = subprocess.run(
            [sys.executable, "-m", "coverage", "report", "--show-missing",
             "--skip-covered", "--include", f"{directory}/*"],
            cwd=ROOT, capture_output=True, text=True,
        )
        text = result.stdout.rstrip()
        if not text:
            print(f"coverage_report: no data for {directory}")
            continue
        print(text)
        print()

    worst = subprocess.run(
        [sys.executable, "-m", "coverage", "report", "--sort=cover"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if worst.returncode == 0:
        lines = worst.stdout.splitlines()
        header = [ln for ln in lines if ln.startswith("TOTAL")]
        print("Least covered modules (where a missing test would hide):")
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[0].endswith(".py") and "%" in parts[2]:
                print(f"  {line.strip()}")
        for line in header:
            print(f"\nTOTAL {line.split('TOTAL', 1)[1].strip()}")
    return 0


def main() -> int:
    data = ROOT / ".coverage"
    if not data.exists():
        run_suite()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())