"""Run every smoke_test_*.py in this directory sequentially and print a
combined summary. Convenience only -- each file remains independently
runnable and independently meaningful; this doesn't replace reading a
given test's own output when something fails, it just saves typing all
the filenames by hand.

Run this directly: `python nodes/smoke_tests/run_all.py`
Filter by substring: `python nodes/smoke_tests/run_all.py memory adafactor`
(runs any smoke_test_*.py whose filename contains any of the given
substrings)

Exits 0 only if every test that ran exited 0.
"""

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def discover_tests(filters: list[str]) -> list[Path]:
    tests = sorted(p for p in _HERE.glob("smoke_test_*.py"))
    if filters:
        tests = [p for p in tests if any(f in p.name for f in filters)]
    return tests


def main():
    filters = sys.argv[1:]
    tests = discover_tests(filters)
    if not tests:
        print(f"No smoke_test_*.py files matched filters {filters!r} in {_HERE}")
        sys.exit(1)

    print(f"Running {len(tests)} test file(s):")
    for t in tests:
        print(f"  {t.name}")

    results: list[tuple[str, int]] = []
    for t in tests:
        print(f"\n{'='*70}\n{t.name}\n{'='*70}")
        proc = subprocess.run([sys.executable, str(t)])
        results.append((t.name, proc.returncode))

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    failed = [name for name, rc in results if rc != 0]
    for name, rc in results:
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {status}: {name}")

    if failed:
        print(f"\n{len(failed)}/{len(results)} test file(s) failed.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} test file(s) passed.")


if __name__ == "__main__":
    main()
