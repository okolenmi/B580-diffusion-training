"""Runs every ``check_*`` script in this folder, each in its own process, and prints one line per script.

    python -m next.tests.run_all             # everything
    python -m next.tests.run_all pipeline    # only scripts whose name contains "pipeline"

The package name is taken from where the file lives, so the same command works wherever the tree is mounted. Exit status
is non-zero when any script fails. ``check_perf`` and the reference equivalences are skipped when their inputs are absent
(they say so themselves).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = __package__ or f"{HERE.parent.name}.tests"


def main(argv: list[str]) -> int:
    wanted = argv[1:]
    names = sorted(p.stem for p in HERE.glob("check_*.py") if not wanted or any(w in p.stem for w in wanted))
    if not names:
        print("no check script matches", wanted)
        return 2
    failed: list[str] = []
    for name in names:
        started = time.perf_counter()
        done = subprocess.run([sys.executable, "-m", f"{PACKAGE}.{name}"], capture_output=True, text=True, cwd=HERE.parent.parent)
        lines = [line for line in (done.stdout + done.stderr).splitlines() if line.strip()]
        spoken = [line for line in done.stdout.splitlines() if line.strip()]  # stderr is only noise next to the summary
        counted = [line for line in spoken if "passed" in line or "PASSED" in line]
        verdict = (counted or spoken or lines or ["(no output)"])[-1].strip()
        ok = done.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {name:<28} {time.perf_counter() - started:5.1f}s  {verdict[:90]}")
        if not ok:
            failed.append(name)
            print("\n".join(f"       {line}" for line in lines[-12:]))
    print(f"\n{len(names) - len(failed)}/{len(names)} scripts passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
