#!/usr/bin/env python3
r"""Compile every .py file in the tree with SyntaxWarning as an error.

The check `check_quality.py` deliberately does not make: it is not a
baseline and it is not style.

**Why not the ruff baseline.** `check_quality.py`'s `RUFF_TARGETS` is
`["backend"]`, and the comment above it says why -- `nodes/` and
`manager/` have conventions of their own (lazy imports inside functions
are load-bearing there) and their several hundred findings would sit in
the baseline forever, "which is the same as having no gate at all".
That reasoning is sound for style, and this check is not style.

An invalid escape sequence in a string literal is never a convention. It
is a file that compiles today with a warning naming the exact line, and
that Python has said it will stop compiling. `nodes/model/tokenizer.py`
carried one for the whole of section 7.3-C2: a module docstring holding
CLIP's splitting pattern `\p{L}|\p{N}|[^\s\p{L}\p{N}]`, not a raw
string. It printed

    nodes/model/tokenizer.py:68: SyntaxWarning: "\\p" is an invalid
    escape sequence. Such sequences will not work in the future.

on import -- and only *sometimes* looked like it, because a populated
`__pycache__` means no recompile and therefore no warning. A fresh clone
warns; the developer's own machine is silent. That asymmetry is the
worst possible shape for a check, and it is why the fix alone is not
enough: nothing would have noticed it coming back.

**Why not `ruff --select W605` instead.** That rule exists and does
catch this (it found all six occurrences ruff reports on that one line,
including the `\s` that a hand-rolled scanner tends to wave through as
"a regex escape"). But `full_gate.sh` skips ruff whenever the gate
interpreter lacks it, and this check needs nothing but the interpreter
it is already running under. A check that can be skipped is a weaker
gate than one that cannot, so this compiles instead -- and compiling
catches the whole language's warning class, not the one rule anyone
thought to select.

**What it costs.** One `compile()` per file, no import, no execution.
The whole tree in well under a second.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories that are not the project's source. `runs/` is gitignored
# measurement output but a run can leave a .py behind; `assets/` is data.
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".mypy_cache", ".ruff_cache", "runs", "assets", "dist", "build",
}


def source_files() -> list[Path]:
    """Every .py under ROOT, in a stable order, as repo-relative paths."""
    found = [
        p for p in ROOT.rglob("*.py")
        if not (SKIP_DIRS & set(p.relative_to(ROOT).parts))
    ]
    return sorted(found, key=lambda p: str(p.relative_to(ROOT)))


def _one_line(text: str) -> str:
    """Flatten a compiler message that spans several lines."""
    return " ".join(text.split())


def main() -> int:
    files = source_files()
    if not files:
        print("check_syntax: no .py files found -- is ROOT right?", file=sys.stderr)
        return 1

    # SyntaxWarning only. DeprecationWarning and friends are somebody
    # else's migration schedule, and turning them into gate failures here
    # would make this the place people go to silence them.
    findings: list[tuple[str, int, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", SyntaxWarning)
        for path in files:
            rel = str(path.relative_to(ROOT))
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                compile(source, rel, "exec", dont_inherit=True, optimize=-1)
            except SyntaxWarning as exc:
                # Python 3.14 promotes the warning to a SyntaxError to get a
                # source line under it, so the caught object's text embeds
                # "SyntaxError:" and a bare newline. Flatten it: the label
                # above already says which of the two this is.
                findings.append((rel, 0, _one_line(f"SyntaxWarning: {exc}")))
            except SyntaxError as exc:
                findings.append((rel, exc.lineno or 0, _one_line(f"SyntaxError: {exc.msg}")))

    for rel, lineno, message in findings:
        where = f"{rel}:{lineno}" if lineno else rel
        print(f"  {where}: {message}")

    print(
        f"check_syntax: {len(files)} file(s) compiled with SyntaxWarning as an "
        f"error, {len(findings)} finding(s)"
    )
    if findings:
        print(
            "  a SyntaxWarning here is a file that will stop compiling; make the "
            "literal raw (r\"\"\" or '), or escape the backslash"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
