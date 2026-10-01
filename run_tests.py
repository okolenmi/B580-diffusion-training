"""Run every smoke test in the repo -- nodes/ and manager/ -- in one
command, with the right interpreter. (server/'s six tests retired with
the archive/ move, M9.)

    python run_tests.py                 # everything (~68 files)
    python run_tests.py memory          # filename-substring filters, same
    python run_tests.py manager         # semantics as nodes/smoke_tests/run_all.py

Exits 0 only if every test that ran exited 0.

Why this exists (and why it picks the interpreter itself): the tests all
import torch, which lives in your ComfyUI venv, not in whatever system
`python` happens to be first on PATH. Running them with the wrong
interpreter doesn't fail cleanly -- it produces a wall of
"ModuleNotFoundError: No module named 'torch'" that reads exactly like a
mass regression (it did, the first time someone ran the suite that way).
So: if the interpreter running this script can already import torch, use
it as-is; otherwise resolve VENV_PYTHON (environment variable, then
.env, then the sibling ../venv/ of the default layout -- the same
fallback run_server.sh and path_tiers.py use) and re-exec the tests
under that. If none works, say exactly what was tried instead of
emitting a wall of identical tracebacks.

This is the "one-line top-level runner" asked for in
docs/review_notes.md; each individual suite's own runner stays
independent and runnable on its own (nodes/smoke_tests/run_all.py has
its own filter semantics worth keeping).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SUITES = ("nodes", "manager")


def _has_torch(python: str) -> bool:
    try:
        return subprocess.run(
            [python, "-c", "import torch"],  # noqa: S603 -- fixed argv, no shell
            capture_output=True, timeout=120,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _venv_python_from_dotenv() -> str | None:
    """Read VENV_PYTHON from .env (repo root) without importing paths.py,
    which would work too but pulls in path auto-detection side effects we
    don't need just to find an interpreter."""
    dotenv = _HERE / ".env"
    if not dotenv.is_file():
        return None
    for line in dotenv.read_text().splitlines():
        line = line.strip()
        if line.startswith("VENV_PYTHON="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _sibling_venv() -> str | None:
    """The ``../venv/bin/python`` of the documented default layout.

    run_server.sh and backend/infrastructure/path_tiers.py both fall back
    to this, so without it the server started fine in a stock three-
    folder checkout while the test runner hard-failed -- the asymmetry
    was a trap in docs/setup.md, which claimed nothing needed
    configuring (fixed 2026-10-01).
    """
    candidate = _HERE.parent / "venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else None


def resolve_interpreter() -> str:
    import os
    if _has_torch(sys.executable):
        return sys.executable
    candidates = (
        os.environ.get("VENV_PYTHON"),
        _venv_python_from_dotenv(),
        _sibling_venv(),
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and _has_torch(candidate):
            return candidate
    tried = "\n".join(f"  - {c}" for c in (sys.executable, *candidates) if c)
    sys.exit(
        "Cannot find a Python interpreter with torch installed.\n"
        f"{tried}\n"
        "Set VENV_PYTHON to your venv's python (see .env.example), or run "
        "this script with that interpreter directly."
    )


def discover_tests(filters: list[str]) -> list[tuple[str, Path]]:
    tests: list[tuple[str, Path]] = []
    for suite in _SUITES:
        suite_dir = _HERE / suite / "smoke_tests"
        found = sorted(suite_dir.glob("smoke_test_*.py"))
        if filters:
            found = [p for p in found if any(f in p.name for f in filters)]
        tests.extend((suite, p) for p in found)
    return tests


def main() -> None:
    filters = sys.argv[1:]
    tests = discover_tests(filters)
    if not tests:
        print(f"No smoke_test_*.py files matched filters {filters!r} under "
              f"{', '.join(_SUITES)}/*/smoke_tests/")
        sys.exit(1)

    python = resolve_interpreter()
    if python != sys.executable:
        print(f"note: {sys.executable} has no torch -- using {python} instead\n")

    print(f"Running {len(tests)} test file(s):")
    for suite, t in tests:
        print(f"  [{suite}] {t.name}")

    results: list[tuple[str, str, int]] = []
    for suite, t in tests:
        print(f"\n{'=' * 70}\n[{suite}] {t.name}\n{'=' * 70}")
        proc = subprocess.run([python, str(t)])  # noqa: S603 -- fixed argv, no shell
        results.append((suite, t.name, proc.returncode))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    failed = [(s, n) for s, n, rc in results if rc != 0]
    for suite, name, rc in results:
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        print(f"  {status}: [{suite}] {name}")

    if failed:
        print(f"\n{len(failed)}/{len(results)} test file(s) failed.")
        sys.exit(1)
    print(f"\nAll {len(results)} test file(s) passed.")


if __name__ == "__main__":
    main()
