#!/usr/bin/env python3
"""Mutation report for the four modules the round-2 review named.

    python scripts/mutation_report.py                 # all four
    python scripts/mutation_report.py supervisor.py   # just one

**Report only.** The point is not "zero survivors" -- it is that every
survivor is either killed by a test or has a written reason it is
equivalent. A mutation that survives tells you a test passes without
checking the thing the test is about; a mutation that is *equivalent*
tells you the code says the same thing twice, which is a different and
milder finding.

Why this drives mutmut's engine by hand
----------------------------------------

mutmut's own runner is pytest. This repository's tests are scripts:
`python backend/tests/test_x.py`, module-level, using a `check()`/`finish()`
pair that collects failures. Importing them into pytest puts their
module-level work at collection time, where a raised assertion is a
collection *error* rather than a test failure -- which works for killing
mutants, and is useless for telling anyone which assertion fired. So this
runs them the way the rest of the repository runs them.

Two details that cost time to find:

* `MetadataWrapper` **deep-copies by default**, so a transformer cannot
  match a mutation's `original_node` by identity -- every one of 57
  mutations came out as a no-op. `unsafe_skip_copy=True` keeps identity
  and all 57 apply.
* `create_mutations` returns the *visited* module, whose nodes are not the
  ones the mutations refer to. Parse the source separately and keep that
  tree, so the transformer matches.

Which tests can kill a mutation
-------------------------------

Measured, not guessed: a targeted coverage pass records which test files
execute lines in each target, and only those are run for its mutants. A
hand-maintained mapping goes stale the first time a test moves, and then
reports a mutant as surviving because nobody ran the test that would have
killed it.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import atexit
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    "backend/application/supervisor.py",
    "backend/infrastructure/jsonl_progress_source.py",
    "backend/application/use_cases/reconcile_runs.py",
    "backend/presentation/sse.py",
]


# --------------------------------------------------------------------------
# Which tests touch which target
# --------------------------------------------------------------------------

def _function_index(path: Path) -> list[tuple[int, int, str]]:
    """(first_line, last_line, name) for every function and method."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, int, str]] = []

    def walk(node, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                out.append((child.lineno, child.end_lineno or child.lineno, name))
                walk(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree)
    return sorted(out)


def _enclosing(index: list[tuple[int, int, str]], line: int) -> tuple[int, str]:
    """(function's last line, name) for the innermost function containing
    `line`, or (line, "<module>") when there is none."""
    best = None
    for first, last, name in index:
        if first <= line <= last and (best is None or (last - first) < (best[0] - best[1])):
            best = (first, last, name)
    return (best[1], best[2]) if best else (line, "<module>")


def _body_lines(path: Path) -> set[int]:
    """Line numbers that live inside a function or method body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.update(range(node.body[0].lineno, node.end_lineno + 1))
    return lines


def _coverage_for_test(job: tuple[str, str]) -> tuple[str, dict[str, list[int]]]:
    """Run one test file under coverage; return its executed lines per file."""
    name, scratch_root = job
    sys.path.insert(0, str(ROOT))
    scratch = Path(scratch_root) / Path(name).stem
    scratch.mkdir(parents=True, exist_ok=True)
    data = Path(scratch_root) / f"{Path(name).stem}.data"
    env = {**os.environ, "TMPDIR": str(scratch)}
    subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--branch",
         f"--data-file={data}", "--source=backend", str(ROOT / "backend" / "tests" / name)],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    if not data.exists():
        return name, {}
    as_json = subprocess.run(
        [sys.executable, "-m", "coverage", "json", f"--data-file={data}", "-o", "-"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if as_json.returncode != 0:
        return name, {}
    try:
        files = json.loads(as_json.stdout).get("files", {})
    except json.JSONDecodeError:
        return name, {}
    return name, {
        path: list(payload.get("executed_lines") or ())
        for path, payload in files.items()
    }


def tests_covering(targets: list[str], jobs: int) -> dict[str, dict[str, set[int]]]:
    """target -> test file -> the lines that test executed in it.

    One coverage run per test file, each into its own `--data-file`, read
    back with `coverage json`, and all of them in parallel -- the pass was
    serial and became the dominant cost once mutation testing itself was
    parallelised, which is the usual way this goes wrong.

    The first version used `--parallel-mode` and read the data with the
    `CoverageData` API; it reported that *nothing* covers supervisor.py,
    when three test files plainly do. A mapping that says "no coverage"
    when there is coverage turns every mutant into a false survivor --
    the worst failure available for a component whose only job is to say
    which tests to run.
    """
    bodies = {t: _body_lines(ROOT / t) for t in targets}
    wanted = {(ROOT / t).resolve() for t in targets}
    found: dict[str, dict[str, set[int]]] = {t: {} for t in targets}

    tests = [p.name for p in sorted((ROOT / "backend" / "tests").glob("test_*.py"))]
    scratch_root = tempfile.mkdtemp(prefix="mutmap-")
    try:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for name, files in pool.map(
                _coverage_for_test,
                [(n, scratch_root) for n in tests],
                chunksize=1,
            ):
                for path, lines in files.items():
                    target = next(
                        (t for t in targets if Path(path).resolve() == (ROOT / t).resolve()),
                        None,
                    )
                    if target is None:
                        continue
                    hit = set(lines) & bodies[target]
                    if hit:
                        found[target][name] = hit
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    _ = wanted  # resolved set kept for clarity of intent above
    return {k: dict(sorted(v.items())) for k, v in found.items()}


def _function_index(path: Path) -> list[tuple[int, int, str]]:
    """(first_line, last_line, name) for every function and method."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: list[tuple[int, int, str]] = []

    def walk(node, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                out.append((child.lineno, child.end_lineno or child.lineno, name))
                walk(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree)
    return sorted(out)


def _enclosing(index: list[tuple[int, int, str]], line: int) -> tuple[int, str]:
    """(function's last line, name) for the innermost function containing
    `line`, or (line, "<module>") when there is none."""
    best = None
    for first, last, name in index:
        if first <= line <= last and (best is None or (last - first) < (best[0] - best[1])):
            best = (first, last, name)
    return (best[1], best[2]) if best else (line, "<module>")


def _body_lines(path: Path) -> set[int]:
    """Line numbers that live inside a function or method body."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.update(range(node.body[0].lineno, node.end_lineno + 1))
    return lines


def mutations_of(path: Path) -> tuple[list, object, str]:
    """(mutations, module, source) for one file, via mutmut's engine."""
    import libcst as cst
    from libcst.metadata import MetadataWrapper
    from mutmut.mutation.mutators import mutation_operators
    from mutmut.mutation.file_mutation import MutationVisitor
    from mutmut.mutation.pragma_handling import get_ignored_lines

    source = path.read_text(encoding="utf-8")
    module = cst.parse_module(source)
    # unsafe_skip_copy: see the module docstring. Without it the metadata
    # wrapper copies the tree and identity matching silently matches
    # nothing.
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    ignored = get_ignored_lines(path.name, source, wrapper)
    visitor = MutationVisitor(mutation_operators, ignored, None)
    wrapper.visit(visitor)
    return visitor.mutations, module, source


def apply_one(mutation, module):
    import libcst as cst

    class ApplyOne(cst.CSTTransformer):
        def on_leave(self, original_node, updated_node):
            if original_node is mutation.original_node:
                return mutation.mutated_node
            return updated_node

    return module.visit(ApplyOne()).code


def _first_changed_line(before: str, after: str) -> int:
    """1-based line number in `before` that the mutation changed."""
    a = before.splitlines()
    b = after.splitlines()
    for index in range(max(len(a), len(b))):
        left = a[index] if index < len(a) else None
        right = b[index] if index < len(b) else None
        if left != right:
            return index + 1
    return 1


def run_tests(names: list[str], timeout: float) -> bool:
    """True if they all pass -- so the mutant survived.

    A hang counts as killed, because a mutation that makes a test stop
    finishing *is* an observable change. But the timeout has to be short:
    an unmutated test file finishes in a couple of seconds, and the first
    full run of this script spent over four minutes on a *single* mutant
    whose effect was to make `test_start_stop.py` wait forever, because
    the default was 300s. At that price a run of a few hundred mutants is
    hours of nothing happening.
    """
    scratch = Path(tempfile.mkdtemp(prefix="mutant-"))
    try:
        for name in names:
            env = {**os.environ, "TMPDIR": str(scratch)}
            try:
                result = subprocess.run(
                    [sys.executable, str(ROOT / "backend" / "tests" / name)],
                    cwd=ROOT, env=env, capture_output=True, text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode != 0:
                return False
        return True
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _restore_on_exit(path: Path, source: str) -> None:
    """Put `path` back on any exit, including SIGTERM/SIGINT."""
    def restore(*_args: object) -> None:
        try:
            if path.read_text(encoding="utf-8") != source:
                path.write_text(source, encoding="utf-8")
        except OSError:
            pass
        if _args:
            raise SystemExit(130)

    atexit.register(restore)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, restore)


def plan_work(
    target: str,
    path: Path,
    mutations: list,
    module,
    source: str,
    index: list[tuple[int, int, str]],
    tests: dict[str, set[int]],
    args: argparse.Namespace,
) -> tuple[list, list[str], int]:
    """(testable jobs, survivors with no possible witness, that count).

    A mutation in a function no test executes cannot be killed by anything,
    so scheduling it would spend a worker to learn nothing. Saying so is
    itself the result, and it is the answer to "why is this uncovered?"
    that a line-coverage number cannot give.
    """
    jobs: list[tuple[str, str, list[str], float]] = []
    survivors: list[str] = []
    unreachable = 0
    for number, mutation in enumerate(mutations):
        if number >= args.max_per_file:
            print(f"  (stopping after {args.max_per_file} of "
                  f"{len(mutations)} mutations -- --max-per-file)")
            break
        try:
            mutated = apply_one(mutation, module)
        except Exception:  # noqa: BLE001 -- cannot even be applied
            continue
        if mutated == source:
            continue
        try:
            compile(mutated, str(path), "exec")
        except SyntaxError:
            continue

        # Only tests that executed the function this mutation sits in can
        # kill it. Running all of them was unusable: 11 test files per
        # mutant over 231 mutants is most of an hour, and ten of those
        # eleven could not observe the change.
        line = _first_changed_line(source, mutated)
        _, where = _enclosing(index, line)
        span = {
            n for first, last, _ in index if first <= line <= last
            for n in range(first, last + 1)
        }
        span.add(line)
        candidates = [name for name, seen in tests.items() if seen & span]
        if not candidates:
            unreachable += 1
            survivors.append(f"line {line} in {where} -- NO TEST EXECUTES IT")
            continue
        jobs.append((target, mutated, candidates, args.timeout))
    return jobs, survivors, unreachable


def measure_file(
    target: str,
    mapping: dict,
    args: argparse.Namespace,
    base: str,
) -> tuple[int, int]:
    """Mutate one file and test every mutation, in parallel.

    Returns (killed, survived). Nothing in ROOT is written: each mutation
    is applied inside a worker's own copy of the repository.
    """
    path = ROOT / target
    print(f"\n=== {target} ===", flush=True)
    try:
        mutations, module, source = mutations_of(path)
    except Exception as exc:  # noqa: BLE001 -- report, do not crash
        print(f"  could not enumerate mutations: {type(exc).__name__}: {exc}")
        return 0, 0

    index = _function_index(path)
    tests: dict[str, set[int]] = mapping.get(target, {})

    if not tests:
        print(f"  {len(mutations)} mutations, but no test covers this file: "
              "every one of them survives by default.")
        return 0, len(mutations)

    jobs, survivors, unreachable = plan_work(
        target, path, mutations, module, source, index, tests, args
    )
    print(f"  {len(mutations)} mutations: {len(jobs)} testable, "
          f"{unreachable} in functions no test executes; "
          f"running on {args.jobs} worker(s)", flush=True)

    killed = 0
    with contextlib.ExitStack() as stack:
        if args.serial:
            results: Iterator[tuple[bool, str]] = (
                _serial_run(job) for job in jobs
            )
        else:
            pool = stack.enter_context(ProcessPoolExecutor(
                max_workers=args.jobs, initializer=_worker_init,
                initargs=(base,),
            ))
            results = pool.map(_worker_run, jobs, chunksize=1)
        for done, (survived_it, diff) in enumerate(results, start=1):
            if survived_it:
                survivors.append(diff)
            else:
                killed += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  ... {done}/{len(jobs)} "
                      f"({killed} killed, {len(survivors)} survived)", flush=True)

    survived = len(survivors)
    tested = killed + survived - unreachable
    pct = (killed / tested * 100) if tested else 0.0
    print(f"  {killed} killed, {survived} survived "
          f"({unreachable} of them unreachable by any test) "
          f"-- {pct:.0f}% of testable mutations killed")
    for entry in survivors:
        print(f"    SURVIVED  {entry[:200]}")
    return killed, survived


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("targets", nargs="*", help="substrings of the path")
    parser.add_argument(
        "--max-per-file", type=int, default=10_000,
        help="stop after this many mutations per file (the full run is slow)",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="seconds per test file; a hang counts as killed",
    )
    parser.add_argument(
        "--jobs", type=int, default=min(8, (os.cpu_count() or 2)),
        help="parallel workers, each with its own copy of the repository",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="run mutants in ROOT instead of in worker copies -- rewrites the "
             "real source files, and exists only to check that the parallel "
             "path gives identical verdicts",
    )
    args = parser.parse_args()

    targets = [
        t for t in TARGETS
        if not args.targets or any(a in t for a in args.targets)
    ]
    if not targets:
        print(f"no target matches {args.targets}; known: {TARGETS}")
        return 1

    print("measuring which tests touch which target ...\n", flush=True)
    mapping = tests_covering(targets, args.jobs)
    for target in targets:
        names = ", ".join(mapping.get(target, {})) or "NO TEST COVERS IT"
        print(f"  {target}: {names}")

    base = tempfile.mkdtemp(prefix="mutation-workers-")
    overall_killed = overall_survived = 0
    try:
        for target in targets:
            killed, survived = measure_file(target, mapping, args, base)
            overall_killed += killed
            overall_survived += survived
    finally:
        shutil.rmtree(base, ignore_errors=True)

    print(f"\nTOTAL: {overall_killed} killed, {overall_survived} survived"
          + (f" ({overall_killed / max(1, overall_killed + overall_survived) * 100:.0f}% killed)"
             if overall_killed + overall_survived else ""))
    print("\nSurvivors are not automatically bugs. Classify each one in "
          "docs/status/mutation-notes.md:\n"
          "  - add a test that kills it, or\n"
          "  - say why the mutant is equivalent (the code says the same "
          "thing twice, the value is unused, ...).")
    return 0


# ==========================================================================
# Parallel execution
# ==========================================================================
#
# The serial version was the bottleneck, and not because the tests are
# slow -- they run in 0.35-0.98s and import no torch. It was that exactly
# one mutant ran at a time, on a machine with six idle cores: measured
# ~5.8s per mutant, of which the single test subprocess was under a
# second. The rest was the serial remainder of every candidate list, plus
# the 20s a hanging mutant costs by design.
#
# Parallelising it the obvious way is blocked by the tool's own worst
# property: every worker would want to rewrite the same source file. The
# fix is also the better design -- **give each worker its own copy of the
# repository and never touch the real one**. Three things follow:
#
#   * workers cannot interfere with each other, or with the working tree;
#   * there is no restore-on-exit to get wrong (it already failed once,
#     leaving a mutant committed);
#   * an interrupted run leaves nothing behind at all.
#
# The copy excludes `runs/` (359 MB of training output) and the other
# data directories, which no unit test reads -- they build their own
# temporaries. Copying the rest is about 3 MB.

_COPY_IGNORE = shutil.ignore_patterns(
    ".git", ".mypy_cache", ".hypothesis", ".pytest_cache", "__pycache__",
    "archive", "runs", "datasets", "models", "node_modules", "mutants",
    ".coverage", ".coverage.*", "frontend", "*.png", "*.safetensors",
)

# Per-process state, set by the pool initialiser.
_WORKER_ROOT: Path | None = None


def _worker_init(base: str) -> None:
    """Build this worker's private copy of the repository, once."""
    global _WORKER_ROOT
    target = Path(base) / f"w{os.getpid()}"
    shutil.copytree(ROOT, target, ignore=_COPY_IGNORE, symlinks=True)
    _WORKER_ROOT = target
    os.chdir(target)


def _serial_run(job: tuple[str, str, list[str], float]) -> tuple[bool, str]:
    """Test one mutant in ROOT itself. Rewrites a real source file."""
    target, mutated, candidates, timeout = job
    path = ROOT / target
    original = path.read_text(encoding="utf-8")
    path.write_text(mutated, encoding="utf-8")
    try:
        survived = _run_tests_in(ROOT, candidates, timeout)
    finally:
        path.write_text(original, encoding="utf-8")
    diff = " / ".join(
        ln for ln in difflib.unified_diff(
            original.splitlines(), mutated.splitlines(), lineterm="", n=0,
        )
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
    )
    return survived, diff[:200]


def _worker_run(job: tuple[str, str, list[str], float]) -> tuple[bool, str]:
    """Test one mutant inside this worker's copy. Never touches ROOT."""
    target, mutated, candidates, timeout = job
    assert _WORKER_ROOT is not None
    path = _WORKER_ROOT / target
    original = path.read_text(encoding="utf-8")
    path.write_text(mutated, encoding="utf-8")
    try:
        survived = _run_tests_in(_WORKER_ROOT, candidates, timeout)
    finally:
        path.write_text(original, encoding="utf-8")
    diff = " / ".join(
        ln for ln in difflib.unified_diff(
            original.splitlines(), mutated.splitlines(), lineterm="", n=0,
        )
        if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
    )
    return survived, diff[:200]


def _run_tests_in(root: Path, names: list[str], timeout: float) -> bool:
    """True if they all pass -- so the mutant survived."""
    scratch = Path(tempfile.mkdtemp(prefix="mutant-"))
    try:
        for name in names:
            env = {**os.environ, "TMPDIR": str(scratch)}
            try:
                result = subprocess.run(
                    [sys.executable, str(root / "backend" / "tests" / name)],
                    cwd=root, env=env, capture_output=True, text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode != 0:
                return False
        return True
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

if __name__ == "__main__":
    raise SystemExit(main())
