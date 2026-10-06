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
* **The exit code**, and the summary lines the gate greps for. A third
  property was added in round 4 and is easy to get wrong: a hermeticity
  self-check runs afterwards, and its failure sets the exit code even when
  every file passed. See `hermeticity_selfcheck()`.

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

#: The two files the hermeticity self-check runs. These are the two that
#: *died* on a checkout with no ComfyUI configured -- "RuntimeError: Cannot
#: find ComfyUI directory" during import, which kills the file rather than
#: failing a check -- in rounds 3 and 4. Both resolve a ComfyUI path at
#: import time, so a regression in the fixture takes them out at once.
HERMETICITY_PROBES = ("test_config.py", "test_settings.py")

#: A COMFY_DIR that cannot exist. See hermeticity_selfcheck().
HERMETICITY_SENTINEL = "/nonexistent/hermeticity-probe/comfy"


def hermeticity_selfcheck(scratch_root: str) -> list[str]:
    """Run two files under ComfyUI settings a fresh clone cannot have.

    Round-4 R4-03, second item: "run_all.py already scrubs the environment
    per file; add a final self-check that runs two representative files
    with no COMFY_DIR anywhere and fails the run if either errors."

    Two variants are run, and the second is the one that actually bites.

    **`COMFY_DIR` removed** is the review's wording and the weaker of the
    two. `.env` is resolved relative to `paths.py`, not to the working
    directory (`paths.py:32`), so on a checkout that has one the file still
    finds a valid ComfyUI whether or not the import-time fixture ran.
    Changing cwd cannot hide `.env`, and moving a developer's `.env` out of
    the tree mid-run is not a thing a test should do to their checkout.

    **`COMFY_DIR` pointing at a path that does not exist** makes the fixture
    observable. Without it a file gets the temp ComfyUI that
    `support.use_temporary_comfy_dir()` installs at import, and with it the
    path resolves to a real directory; a file that somehow missed the
    fixture would fall through to `get_comfy_dir()` and raise, which is
    precisely the failure the fixture exists to prevent. Both halves of
    that are measured, not assumed:

        no fixture   -> RuntimeError: Cannot find ComfyUI directory
        with fixture -> /tmp/backend-suite-comfy-XXXX, exists: True

    So the second variant fails where the first cannot, and costs one extra
    subprocess per file. `scripts/check_bare_checkout.sh` is the version
    that gets it right outright, by running a `git archive` of HEAD with no
    `.env` in it at all; that is the thorough check, and it is too slow for
    every gate run, which is what this one is for.

    Returns a list of failure descriptions -- empty means it passed. It
    raises nothing: a self-check that crashes the run tells you less than
    one that reports why it could not run.
    """
    failures: list[str] = []

    if Path(HERMETICITY_SENTINEL).exists():
        return [f"the sentinel path {HERMETICITY_SENTINEL} exists, so the "
                f"variant that makes the fixture observable would pass "
                f"vacuously -- pick another HERMETICITY_SENTINEL"]

    variants = (
        ("COMFY_DIR removed", None),
        ("COMFY_DIR points at a path that does not exist",
         HERMETICITY_SENTINEL),
    )

    for name in HERMETICITY_PROBES:
        for label, value in variants:
            env = {**os.environ, "TMPDIR": scratch_root}
            env.pop("COMFY_DIR", None)
            if value is not None:
                env["COMFY_DIR"] = value
            try:
                result = subprocess.run(
                    [sys.executable, str(HERE / name)],
                    capture_output=True, text=True, env=env, timeout=600,
                    cwd=HERE.parent.parent,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{name}, {label}: TIMED OUT after 600s")
                continue

            if result.returncode:
                detail = ""
                for line in (result.stdout or result.stderr or "").splitlines():
                    if "ComfyUI" in line or "Error" in line:
                        detail = line.strip()[:150]
                        break
                failures.append(
                    f"{name}, {label}: exit {result.returncode}"
                    + (f" -- {detail}" if detail else "")
                )
            else:
                verdict = next(
                    (line.strip() for line in result.stdout.splitlines()
                     if line.startswith("SMOKE TEST:")), "")
                print(f"  ok    {name:18} {label:52} {verdict}")

    return failures


def report(total: int, failed: list[str], unhermetic: list[str]) -> int:
    """Print the summary the gate greps for. Returns the exit code.

    Split out of main() so that adding the hermeticity self-check did not
    push main() past the complexity limit -- the alternative was raising the
    ruff baseline, and this project's rule is that the baseline never goes
    up.
    """
    print("\n" + "=" * 60)
    if unhermetic:
        print(f"HERMETICITY SELF-CHECK: {len(unhermetic)} of "
              f"{len(HERMETICITY_PROBES) * 2} check(s) failed")
        for line in unhermetic:
            print(f"  - {line}")
        print("  These files pass on a configured machine. A failure here "
              "means the suite only passes because of this checkout's .env; "
              "see hermeticity_selfcheck() in this file, and "
              "scripts/check_bare_checkout.sh for the thorough version.")

    if failed:
        print(f"BACKEND TESTS: {len(failed)}/{total} FILE(S) FAILED")
        for name in failed:
            print(f"  - {name}")
        # Point at the kept output rather than at "run it again": a
        # load-sensitive flake will usually pass on the second attempt,
        # and a second attempt is evidence about the second attempt.
        print(f"\n  each failing file's output: {log_dir()}/<name>.log")
    else:
        print(f"BACKEND TESTS: ALL {total} FILE(S) PASSED")

    # A hermeticity failure fails the run even when every file passed.
    # Reporting "ALL PASSED" and exiting 0 above a red self-check would let
    # the gate go green on a suite that does not work on a fresh clone,
    # which is the whole thing the self-check exists to catch.
    return 1 if (failed or unhermetic) else 0


def log_dir() -> Path:
    """Where the last run's per-file output is kept.

    Only failures are written. A passing file's output is printed by
    main() and then has no further use, while a failing file's output is
    the only evidence of what went wrong -- and several of these files
    fail only under load, so the natural response to a flake (run it
    again) describes a *new* run rather than the failed one, and usually
    passes, which is precisely when the original output matters most.

    Overwritten per file on each run, so the directory holds one run's
    worth and cannot grow: the scratch-directory leak fixed in
    docs/known-issues/resolved.md (4,834 directories, 2.1 GB of /tmp
    tmpfs) is the same failure mode in miniature, and this must not
    reintroduce it.
    """
    d = Path(tempfile.gettempdir()) / "backend-test-logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _keep_output(name: str, rc: int, output: str) -> None:
    """Persist one failing file's output. Never raises.

    A log that cannot be written must not turn a passing run into a
    failing one, or a full disk into a red gate; the verdict comes from
    the exit code and nothing here may touch it.
    """
    if rc == 0:
        return
    try:
        (log_dir() / f"{name}.log").write_text(
            f"exit {rc}\n\n{output}", errors="replace")
    except OSError:
        pass


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
        output = (result.stdout or "") + (result.stderr or "")
        _keep_output(name, result.returncode, output)
        return name, result.returncode, output
    except subprocess.TimeoutExpired:
        output = f"TIMED OUT after 600s: {name}"
        _keep_output(name, 124, output)
        return name, 124, output
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
    unhermetic: list[str] = []
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

        # The hermeticity self-check runs last, and separately from the
        # files above, because it is a different question: not "do these
        # files pass" but "would they pass on a machine that has never heard
        # of the developer". A file can pass either way.
        print("\n=== hermeticity self-check ===")
        if args.only:
            print("  skipped: --only was given, so these files were not "
                  "under test")
        else:
            unhermetic = hermeticity_selfcheck(scratch_root)
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)

    return report(len(test_files), failed, unhermetic)


if __name__ == "__main__":
    raise SystemExit(main())