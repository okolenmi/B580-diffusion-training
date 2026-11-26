"""All browser checks, one after another, each in its own process (a failed one cannot leave a server or a page to the next).

    python -m tests.browser                # every check
    python -m tests.browser apps runs      # only these (names of the check_<name>.py files)
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main(argv: list[str]) -> int:
    names = sorted(p.stem.removeprefix("check_") for p in HERE.glob("check_*.py"))
    chosen = argv or names
    unknown = [n for n in chosen if n not in names]
    if unknown:
        print(f"unknown check {unknown}; choose from {', '.join(names)}")
        return 2
    results = []
    for name in chosen:
        print(f"\n===== browser: {name}", flush=True)
        began = time.monotonic()
        code = subprocess.run([sys.executable, "-m", f"tests.browser.check_{name}"], cwd=ROOT).returncode
        results.append((name, code, time.monotonic() - began))
    print("\nbrowser checks:")
    for name, code, seconds in results:
        print(f"  {'ok  ' if code == 0 else 'FAIL'} {name:<12} {seconds:5.0f}s")
    return int(any(code for _, code, _ in results))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
