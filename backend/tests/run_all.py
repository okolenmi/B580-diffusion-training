"""Run every backend test file in a fresh interpreter.

    python backend/tests/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    here = Path(__file__).resolve().parent
    test_files = sorted(here.glob("test_*.py"))
    if not test_files:
        print("no test files found")
        return 1

    failed: list[str] = []
    for path in test_files:
        print(f"\n=== {path.name} ===")
        result = subprocess.run([sys.executable, str(path)])
        if result.returncode != 0:
            failed.append(path.name)

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
