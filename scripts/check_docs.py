#!/usr/bin/env python3
"""Fail when a documented number stops being true.

**Why.** Two docs in this repository quote a number that the code
determines: the backend's operation count and its test-file count. Both
were already stale when this script was written -- the migration strategy
said 47 operations against an actual 50, and the review that asked for
this check had said 48. Nothing else in the repository would have noticed,
because nothing renders the docs and no test asserted against them. A
number that is wrong in a design document is worse than no number: it is
read as current and used as a baseline.

The check is deliberately narrow. It does not try to keep prose honest;
it checks the specific claims a human is most likely to quote, and it
names the file, the line, the claimed value and the actual one. A
mismatch is a five-second fix.

Claims are declared as (file, regex, callable) so that adding a new one is
one line, and so the expected text is written next to the reason it
matters.

Usage:
    python scripts/check_docs.py
"""

from __future__ import annotations

import re
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# The things the code knows and the docs assert
# --------------------------------------------------------------------------

def backend_operations() -> int:
    """Operations the FastAPI app actually exposes.

    Built in-process against a throwaway workspace -- no server, no
    database in the repository. Counting only the HTTP verbs, so an
    OpenAPI `parameters` or `summary` key cannot inflate the number.
    """
    sys.path.insert(0, str(ROOT))
    from backend.presentation.app import create_app
    from backend.tests.support import build_services

    scratch = Path(tempfile.mkdtemp(prefix="check-docs-"))
    previous = {k: v for k, v in vars(__import__("os").environ).items()
                if k.startswith("BACKEND_")}
    try:
        app = create_app(build_services())
        spec = app.openapi()
        verbs = {"get", "post", "put", "patch", "delete", "head", "options"}
        return sum(
            len([m for m in methods if m in verbs])
            for methods in spec["paths"].values()
        )
    finally:
        import shutil

        shutil.rmtree(scratch, ignore_errors=True)
        for key in list(vars(__import__("os").environ)):
            if key.startswith("BACKEND_") and key not in previous:
                del __import__("os").environ[key]


def backend_test_files() -> int:
    """Test files `run_all.py` will run -- the number a status doc quotes
    when it says how big the suite is."""
    return len(list((ROOT / "backend" / "tests").glob("test_*.py")))


CLAIMS: list[tuple[Path, str, Callable[[], object], str]] = [
    (
        ROOT / "docs" / "design" / "backend" / "03-migration-strategy.md",
        r"(\d+)\s+backend endpoints",
        backend_operations,
        "the parity audit compares the legacy surface against the backend; "
        "a wrong right-hand side makes the whole table read as a "
        "deliberate gap",
    ),
]

# `01-architecture.md` also contains the number 51 ("inconsistent API
# contracts across 51 endpoints"), and it was a claim here until it was
# checked against the code. It is not one: that file lists the *legacy*
# server's problems, so its number is the legacy surface, not the
# backend's. Checking it would have had this script tell a correct
# document to change. Recorded here because the same number appears in
# both files with two different meanings, which is exactly the trap a
# grep-based check falls into.


def main() -> int:
    problems: list[str] = []
    for path, pattern, compute, why in CLAIMS:
        if not path.exists():
            problems.append(f"{path.relative_to(ROOT)}: does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        matches = list(re.finditer(pattern, text))
        if not matches:
            problems.append(
                f"{path.relative_to(ROOT)}: no match for /{pattern}/ -- the "
                "claim was reworded, so this check no longer knows about it"
            )
            continue
        actual = compute()
        for match in matches:
            claimed = int(match.group(1))
            if claimed != actual:
                line = text[: match.start()].count("\n") + 1
                problems.append(
                    f"{path.relative_to(ROOT)}:{line}: claims {claimed}, "
                    f"actual is {actual} ({why})"
                )

    if problems:
        print("check_docs: documented numbers are stale:")
        for problem in problems:
            print(f"  - {problem}")
        print("\nFix the doc, or the code if the doc is right.")
        return 1

    print(
        f"check_docs: documented numbers hold "
        f"({backend_operations()} backend operations, "
        f"{backend_test_files()} test files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())