"""Run every backend test file in a fresh interpreter.

    python backend/tests/run_all.py

**Each file gets its own TMPDIR, and it is removed afterwards.**

The test files create scratch directories with ``tempfile.mkdtemp``, which
returns a name and hands back no handle -- nothing ever removes them, in
any of the 25 files. One full suite run leaves roughly 40 of them behind,
and the suite is run many times, so this accumulated to over four thousand
directories and two gigabytes of ``/tmp`` on the machine that found it.
``/tmp`` is a tmpfs here, so it is not disk that fills up but RAM: a
suitably unlucky moment is all it takes for a test to fail because a
scratch directory could not be created, and the failure names a test that
has nothing wrong with it.

Setting ``TMPDIR`` per file is the smallest change that actually holds:
``tempfile`` honours it, so every scratch path the file creates lands
inside a directory this script owns and then deletes. A test that leaks
now leaks into its own TMPDIR and nowhere else, and one file's litter
cannot be mistaken for another's.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    test_files = sorted(here.glob("test_*.py"))
    if not test_files:
        print("no test files found")
        return 1

    base = Path(tempfile.mkdtemp(prefix="backend-testrun-"))
    failed: list[str] = []
    try:
        for path in test_files:
            print(f"\n=== {path.name} ===")
            # One TMPDIR per file, not one for the run: a file that leaks
            # then cannot influence the next file's scratch paths, and the
            # deletion below is scoped to the file that made the mess.
            scratch = base / path.stem
            scratch.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "TMPDIR": str(scratch)}
            try:
                result = subprocess.run([sys.executable, str(path)], env=env)
                if result.returncode != 0:
                    failed.append(path.name)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)

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