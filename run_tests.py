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

import concurrent.futures as cf
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SUITES = ("nodes", "manager")

#: Extra subdirectories of `<suite>/smoke_tests/` to search, per suite.
#:
#: `nodes/smoke_tests/gpu/` holds the tests that need the accelerator.
#: Without this the gate would run none of them, since they are no longer
#: directly in `smoke_tests/`. A missing directory is not an error, so this
#: also works on a checkout from before the move.
_EXTRA_DIRS: dict[str, tuple[str, ...]] = {"nodes": ("gpu",)}


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


def discover_tests(filters: list[str], include_gpu: bool = True
                   ) -> list[tuple[str, Path]]:
    tests: list[tuple[str, Path]] = []
    for suite in _SUITES:
        suite_dir = _HERE / suite / "smoke_tests"
        found = sorted(suite_dir.glob("smoke_test_*.py"))
        for extra in _EXTRA_DIRS.get(suite, ()):
            if not include_gpu and extra == "gpu":
                continue
            extra_dir = suite_dir / extra
            if extra_dir.is_dir():
                found += sorted(extra_dir.glob("smoke_test_*.py"))
        if filters:
            found = [p for p in found if any(f in p.name for f in filters)]
        tests.extend((suite, p) for p in found)
    return tests


def _is_gpu(test: tuple[str, Path]) -> bool:
    return "smoke_tests/gpu" in test[1].as_posix()


def _banner(suite: str, name: str) -> None:
    print(f"\n{'=' * 70}\n[{suite}] {name}\n{'=' * 70}", flush=True)


def _last_run_dir() -> Path:
    """Where the previous run's per-file output is kept.

    A failing run's output is printed, but printing is not the same as
    keeping it: this suite has files that fail only under load (see
    docs/known-issues/pending-testing.md's cooperative-stop entry, and
    the same class of flake fixed at ddd04df), and the natural response
    to a flake is to run the file again -- which tells you about *now*,
    not about the run that failed, and costs the whole suite's wall clock
    each time.

    So every file's output is written here as it completes, one file per
    log, overwritten at the start of each run. Overwritten rather than
    appended so this cannot grow without bound: the leak fix in
    docs/known-issues/resolved.md removed thousands of scratch
    directories from /tmp, and a log directory that accumulated a full
    suite's output per run would reintroduce the same shape of problem
    in a smaller way. Only the most recent run is on disk, which is the
    only one anyone can act on.
    """
    d = Path(tempfile.gettempdir()) / "b580-test-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_one(job: tuple[str, str, Path, str]
             ) -> tuple[str, str, int, float, str]:
    suite, name, path, interpreter = job
    started = time.monotonic()
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        [interpreter, str(path)], capture_output=True, text=True)
    elapsed = time.monotonic() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    try:
        (_last_run_dir() / f"{suite}__{name}.log").write_text(output, errors="replace")
    except OSError:
        # A log that cannot be written must not fail a run: the verdict
        # comes from the exit code, which is unaffected. Losing the log
        # costs diagnosability, not correctness.
        pass
    return (suite, name, proc.returncode, elapsed, output)


def _default_jobs() -> int:
    """Enough workers to use the machine, leaving one core for the parent.

    Four is the default rather than "one per core" because the heaviest
    files are torch-heavy and oversubscribing them makes the wall clock
    worse, not better; measured on this six-core box, three workers already
    reach the floor set by the serial GPU half.
    """
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def main() -> None:
    argv = sys.argv[1:]
    jobs = _default_jobs()
    serial = False
    include_gpu = True

    # Flags are stripped before the rest is treated as filename filters, so
    # `run_tests.py memory` keeps meaning exactly what it meant before.
    filters: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--serial":
            serial = True
        elif argument == "--no-gpu":
            include_gpu = False
        elif argument == "--jobs" and index + 1 < len(argv):
            index += 1
            jobs = max(1, int(argv[index]))
        elif argument.startswith("--jobs="):
            jobs = max(1, int(argument.split("=", 1)[1]))
        elif argument in ("-h", "--help"):
            print(__doc__)
            print("    --jobs N     parallel workers for the non-GPU files "
                  f"(default {_default_jobs()})\n"
                  "    --serial     one file at a time, as this always was\n"
                  "    --no-gpu     skip nodes/smoke_tests/gpu/\n")
            sys.exit(0)
        else:
            filters.append(argument)
        index += 1

    tests = discover_tests(filters, include_gpu=include_gpu)
    if not tests:
        print(f"No smoke_test_*.py files matched filters {filters!r} under "
              f"{', '.join(_SUITES)}/*/smoke_tests/ (and any extra dirs)")
        sys.exit(1)

    python = resolve_interpreter()
    if python != sys.executable:
        print(f"note: {sys.executable} has no torch -- using {python} instead\n")
    # Everything below runs the resolved interpreter, not this process's.
    interpreter = python

    print(f"Running {len(tests)} test file(s):")
    for suite, t in tests:
        print(f"  [{suite}] {t.name}")

    gpu = [t for t in tests if _is_gpu(t)]
    cpu = [t for t in tests if not _is_gpu(t)]

    results: list[tuple[str, str, int, float]] = []

    # The GPU half first, alone and one at a time.
    #
    # There is one card, and these files each load a real multi-gigabyte
    # checkpoint into it. Two at once is not slower, it is *wrong*: they OOM
    # each other and a failure then means contention rather than a defect.
    # Keeping them serial is the whole reason `[nodes]` and `gpu/` failures
    # can be read as real.
    for suite, t in gpu:
        _banner(suite, t.name)
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            [interpreter, str(t)])
        results.append((suite, t.name, proc.returncode, 0.0))

    # The CPU half in a pool. Threads rather than processes because each
    # job is a `subprocess.run` that just waits on a child; there is nothing
    # to parallelise in this process.
    if serial or jobs == 1:
        for suite, t in cpu:
            _banner(suite, t.name)
            proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
                [interpreter, str(t)])
            results.append((suite, t.name, proc.returncode, 0.0))
    else:
        with cf.ThreadPoolExecutor(max_workers=min(jobs, len(cpu) or 1)) as pool:
            futures = {
                pool.submit(_run_one, (suite, t.name, t, interpreter))
                for suite, t in cpu
            }
            for future in cf.as_completed(futures):
                suite, name, rc, elapsed, output = future.result()
                _banner(suite, name)
                # Printed whether it passed or failed: a passing run's output
                # is the evidence, and suppressing it would make a green gate
                # unreadable.
                sys.stdout.write(output)
                sys.stdout.flush()
                results.append((suite, name, rc, elapsed))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    failed = [(s, n) for s, n, rc, _ in results if rc != 0]
    for suite, name, rc, elapsed in results:
        status = "PASS" if rc == 0 else f"FAIL (exit {rc})"
        timing = f"  {elapsed:6.1f}s" if elapsed else ""
        print(f"  {status}: [{suite}] {name}{timing}")
    slowest = sorted((e, s, n) for s, n, rc, e in results if e)
    if slowest:
        print("\n  slowest: " + ", ".join(
            f"{n} {e:.1f}s" for e, _s, n in reversed(slowest[-5:])))
        print(f"  {len(gpu)} GPU file(s) ran serially; the CPU half used "
              f"{'--serial' if serial else min(jobs, len(cpu) or 1)} worker(s)")

    if failed:
        print(f"\n{len(failed)}/{len(results)} test file(s) failed.")
        # Name the log for each one, so the failing output is readable
        # without re-running anything. Re-running answers a different
        # question -- it describes a new run, not the one that failed --
        # and for a load-sensitive flake it usually passes, which is
        # exactly when the original output is the only evidence there is.
        for suite, name in failed:
            print(f"    {_last_run_dir() / f'{suite}__{name}.log'}")
        sys.exit(1)
    print(f"\nAll {len(results)} test file(s) passed.")


if __name__ == "__main__":
    main()
