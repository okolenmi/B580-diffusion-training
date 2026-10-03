#!/usr/bin/env python3
"""Fail a test file that defines a test it never runs.

    python scripts/check_test_wiring.py

**Why this exists.** `backend/tests/support.py::finish()` now refuses to
report success for a file that ran zero checks, which catches the crudest
version of this. It cannot catch a file that runs *some* checks and
silently orphans the rest -- and this repository has had that twice: a
restructure left the last call to `finish()` above the code, or a `main()`
stopped calling a function it still defined, and the suite stayed green.

The runtime count is the backstop; this is the thing that tells you *which*
test was orphaned, before it runs.

**The rule.** A function named ``test_*`` (or ``check_*``) defined at
module level must be referenced somewhere else in its own file. Being
defined is not being run.

Deliberately conservative about what counts as a reference:

* any call, attribute access or bare name counts, so a name in a
  ``tests = [...]`` list, a ``main()`` that calls it, and a wrapper that
  forwards to it all satisfy this;
* a definition in a ``class`` is skipped -- those are helpers, and the
  repository's test files use the prefix on module-level functions only;
* a ``test_*`` name that is only ever *defined* is the finding.

Exits non-zero with one line per orphan.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Where the project's tests live. Anything matching these is scanned.
TEST_GLOBS = ("backend/tests/test_*.py", "nodes/smoke_tests/smoke_test_*.py",
              "manager/smoke_tests/smoke_test_*.py")

#: Prefixes that mean "this is a test" at module level.
TEST_PREFIXES = ("test_", "check_")


def defined_tests(tree: ast.Module) -> list[str]:
    """Module-level functions whose name marks them as a test."""
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith(TEST_PREFIXES)
    ]


def referenced_names(tree: ast.Module) -> set[str]:
    """Every identifier the module mentions.

    Note there is no "minus the definitions" step, which is the obvious
    thing to add and is wrong: a `def` statement does not produce an
    `ast.Name`, so a definition's name never appears as a reference in the
    first place. Subtracting the defined names from the seen names deletes
    the *call sites* too, which reported 413 orphans in a repository where
    every one of them is called from a `main()`.
    """
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            seen.add(node.id)
        elif isinstance(node, ast.Attribute):
            seen.add(node.attr)
        elif isinstance(node, ast.alias):
            # `from x import test_y as z` still mentions test_y.
            seen.add(node.asname or node.name)
    return seen


def scan(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path.relative_to(ROOT)}: does not parse ({exc})"]
    referenced = referenced_names(tree)
    findings = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith(TEST_PREFIXES):
            continue
        if node.name in referenced:
            continue
        findings.append(
            f"{path.relative_to(ROOT)}:{node.lineno}: {node.name}() is "
            f"defined but never referenced -- it will not run"
        )
    return findings


def main() -> int:
    files = sorted(
        p for pattern in TEST_GLOBS for p in ROOT.glob(pattern) if p.is_file()
    )
    if not files:
        print(f"check_test_wiring: no test files matched {TEST_GLOBS}")
        return 1

    problems: list[str] = []
    defined = 0
    for path in files:
        defined += len([
            n for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name.startswith(TEST_PREFIXES)
        ])
        problems += scan(path)

    if problems:
        print("check_test_wiring: tests that are defined but never run:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            f"\n{len(problems)} orphan(s) across {len(files)} files "
            f"({defined} test functions defined)."
        )
        print(
            "A defined test proves nothing. Call it, or delete it -- an\n"
            "uncalled one is a comment that looks like a safety net."
        )
        return 1

    print(
        f"check_test_wiring: {defined} test functions across {len(files)} "
        f"files, all reachable"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())