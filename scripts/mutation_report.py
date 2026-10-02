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
import os
import shutil
import signal
import subprocess
import sys
import tempfile
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


def tests_covering(targets: list[str]) -> dict[str, list[str]]:
    """target -> test files that execute at least one line of it.

    One coverage run per test file, each into its own `--data-file`, read
    back with `coverage json`. The first version used `--parallel-mode`
    and read the data with the `coverage.CoverageData` API; it reported
    that *nothing* covers supervisor.py, which three test files plainly
    do -- so a silent "no coverage" was being taken as a real answer,
    which is the worst possible failure for a mapping whose whole job is
    to say which tests to run.
    """
    bodies = {t: _body_lines(ROOT / t) for t in targets}
    # target -> test file -> the lines that test executed in it. Kept as
    # line sets rather than a yes/no, because the next step needs them.
    found: dict[str, dict[str, set[int]]] = {t: {} for t in targets}
    tests = sorted((ROOT / "backend" / "tests").glob("test_*.py"))
    scratch_root = Path(tempfile.mkdtemp(prefix="mutmap-"))
    try:
        for test in tests:
            scratch = scratch_root / test.stem
            scratch.mkdir(parents=True, exist_ok=True)
            data = scratch_root / f"{test.stem}.data"
            env = {**os.environ, "TMPDIR": str(scratch)}
            subprocess.run(
                [sys.executable, "-m", "coverage", "run", "--branch",
                 f"--data-file={data}", "--source=backend", str(test)],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            if not data.exists():
                continue
            as_json = subprocess.run(
                [sys.executable, "-m", "coverage", "json",
                 f"--data-file={data}", "-o", "-"],
                cwd=ROOT, capture_output=True, text=True,
            )
            if as_json.returncode != 0:
                continue
            try:
                files = json.loads(as_json.stdout).get("files", {})
            except json.JSONDecodeError:
                continue
            for target in targets:
                for name, payload in files.items():
                    if Path(name).resolve() != (ROOT / target).resolve():
                        continue
                    # Inside a function body, not merely imported. Matching
                    # on any executed line said that 25 of 26 test files
                    # covered supervisor.py, because importing it runs its
                    # module-level lines -- and "runs the import" cannot
                    # kill a mutation inside `_guard`.
                    hit = set(payload.get("executed_lines") or ()) & bodies[target]
                    if hit:
                        found[target][test.name] = hit
    finally:
        shutil.rmtree(scratch_root, ignore_errors=True)
    return {k: dict(sorted(v.items())) for k, v in found.items()}


# --------------------------------------------------------------------------
# Mutation
# --------------------------------------------------------------------------

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


def measure_file(
    target: str, mapping: dict, args: argparse.Namespace
) -> tuple[int, int]:
    """Mutate one file, run the tests that can see each change.

    Returns (killed, survived). Owns restoring the file: it rewrites the
    source in place, and a tool that leaves a mutant behind is worse than
    no tool.
    """
    path = ROOT / target
    source_backup = path.read_text(encoding="utf-8")
    print(f"\n=== {target} ===")
    try:
        mutations, module, source = mutations_of(path)
    except Exception as exc:  # noqa: BLE001 -- report, do not crash
        print(f"  could not enumerate mutations: {type(exc).__name__}: {exc}")
        return 0, 0

    index = _function_index(path)
    tests: dict[str, set[int]] = mapping.get(target, {})

    # Restore the file on any exit, including a signal. Being killed
    # partway through used to leave a mutant committed to the working
    # tree, which happened.
    _restore_on_exit(path, source_backup)

    if not tests:
        print(f"  {len(mutations)} mutations, but no test covers this file: "
              "every one of them survives by default.")
        return 0, len(mutations)

    killed = survived = 0
    survivors: list[str] = []
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
        # kill it. Running all of them worked and was unusable: 11 test
        # files per mutant over 231 mutants is most of an hour, and ten of
        # those eleven could not observe the change.
        line = _first_changed_line(source, mutated)
        _, where = _enclosing(index, line)
        span = set(range(line, line + 1)) | {
            n for first, last, _ in index if first <= line <= last
            for n in range(first, last + 1)
        }
        candidates = [name for name, seen in tests.items() if seen & span]
        if not candidates:
            # No test runs this function at all: the mutant survives for a
            # reason worth knowing, not for want of trying.
            survived += 1
            survivors.append(f"line {line} in {where} -- NO TEST EXECUTES IT")
            continue

        path.write_text(mutated, encoding="utf-8")
        try:
            survived_it = run_tests(candidates, args.timeout)
        finally:
            path.write_text(source_backup, encoding="utf-8")

        if (number + 1) % 25 == 0 or number + 1 == len(mutations):
            print(f"  ... {number + 1}/{len(mutations)} "
                  f"({killed} killed, {survived} survived)", flush=True)
        if survived_it:
            survived += 1
            diff = [
                ln for ln in difflib.unified_diff(
                    source.splitlines(), mutated.splitlines(),
                    lineterm="", n=0,
                )
                if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))
            ]
            survivors.append(" / ".join(diff[:4]))
        else:
            killed += 1

    total = killed + survived
    pct = (killed / total * 100) if total else 0.0
    print(f"  {len(mutations)} mutations: {killed} killed, "
          f"{survived} survived ({pct:.0f}% killed)")
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
    args = parser.parse_args()

    targets = [
        t for t in TARGETS
        if not args.targets or any(a in t for a in args.targets)
    ]
    if not targets:
        print(f"no target matches {args.targets}; known: {TARGETS}")
        return 1

    print("measuring which tests touch which target ...\n", flush=True)
    mapping = tests_covering(targets)
    for target in targets:
        names = ", ".join(mapping.get(target, {})) or "NO TEST COVERS IT"
        print(f"  {target}: {names}")

    overall_killed = overall_survived = 0
    for target in targets:
        killed, survived = measure_file(target, mapping, args)
        overall_killed += killed
        overall_survived += survived

    print(f"\nTOTAL: {overall_killed} killed, {overall_survived} survived"
          + (f" ({overall_killed / max(1, overall_killed + overall_survived) * 100:.0f}% killed)"
             if overall_killed + overall_survived else ""))
    print("\nSurvivors are not automatically bugs. Classify each one in "
          "docs/status/mutation-notes.md:\n"
          "  - add a test that kills it, or\n"
          "  - say why the mutant is equivalent (the code says the same "
          "thing twice, the value is unused, ...).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())