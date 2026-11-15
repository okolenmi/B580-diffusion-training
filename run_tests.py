"""Everything that can be checked without the card, in one command.

    python run_tests.py                      # all four suites
    python run_tests.py trainer backend      # only these suites
    python run_tests.py trainer -- memory    # name parts after -- go to the trainer checks

Suites: ``imports`` (layer rules and the ComfyUI guard), ``trainer`` (``tests/check_*.py``), ``backend``
(``backend/tests``) and ``frontend`` (``node --test``). The Python suites need torch, so they run with the first
interpreter that has it: this one, then ``$VENV_PYTHON``, then ``VENV_PYTHON`` from ``.env``, then ``../venv``.
Hardware checks are not here; they are listed in TESTING.md.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def has_torch(python: str) -> bool:
    try:
        return subprocess.run([python, "-c", "import torch"], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def dotenv_python() -> str | None:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf8").splitlines():
        key, _, value = line.strip().partition("=")
        if key == "VENV_PYTHON" and value.strip():
            return value.strip().strip("\"'")
    return None


def resolve_interpreter() -> str:
    sibling = ROOT.parent / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    candidates = [os.environ.get("VENV_PYTHON"), dotenv_python(), str(sibling)]
    if has_torch(sys.executable):
        return sys.executable
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and has_torch(candidate):
            return candidate
    tried = "\n".join(f"  - {c}" for c in (sys.executable, *candidates) if c)
    sys.exit(f"No Python with torch found.\n{tried}\nSet VENV_PYTHON (see .env.example) or run this with that interpreter.")


def suites(python: str, trainer_filters: list[str]) -> dict[str, list[str] | None]:
    frontend = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "frontend" / "tests").glob("*.test.mjs"))
    return {
        "imports": [python, "scripts/check_imports.py"],
        "trainer": [python, "-m", "tests.run_all", *trainer_filters],
        "backend": [python, "backend/tests/run_all.py"],
        "frontend": ["node", "--test", *frontend] if shutil.which("node") else None,
    }


def main(argv: list[str]) -> int:
    filters: list[str] = []
    if "--" in argv:
        cut = argv.index("--")
        argv, filters = argv[:cut], argv[cut + 1:]
    python = resolve_interpreter()
    table = suites(python, filters)
    unknown = [a for a in argv if a not in table]
    if unknown:
        print(f"unknown suite {unknown}; choose from {', '.join(table)}")
        return 2
    print(f"interpreter: {python}")
    results: list[tuple[str, str, float]] = []
    for name in argv or list(table):
        command = table[name]
        if command is None:
            results.append((name, "skipped (node is not installed)", 0.0))
            continue
        started = time.perf_counter()
        print(f"\n== {name} ==", flush=True)
        code = subprocess.run(command, cwd=ROOT).returncode
        results.append((name, "ok" if code == 0 else f"FAILED (exit {code})", time.perf_counter() - started))
    print("\n" + "\n".join(f"{name:10} {verdict:30} {seconds:6.1f}s" for name, verdict, seconds in results))
    return 0 if all(v == "ok" or v.startswith("skipped") for _, v, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
