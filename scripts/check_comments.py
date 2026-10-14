"""Comment-policy report for the rework tree (``next/``).

    python scripts/check_comments.py           # report, exit 0
    python scripts/check_comments.py --strict  # exit 1 when anything is flagged

Flags (rework/80-developer-guide.md section 12): a comment or docstring line that holds a
measurement (a number with a unit) or history wording. They belong in
``docs/known-issues`` or git history. Also reports each file's comment share, so a file
that drifts toward the old 40 percent is visible.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "next"
MEASURE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:MB|GB|MiB|GiB|ms|it/s|steps?/s|s/step|%)(?![A-Za-z])")
HISTORY = re.compile(r"\b(previously|used to|no longer|after the fix|was changed|legacy|formerly|since the rewrite)\b", re.I)
SHARE_LIMIT = 0.25


def comment_lines(source: str) -> dict[int, str]:
    lines = source.splitlines()
    found: dict[int, str] = {}
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            found[i] = stripped
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                for ln in range(body[0].lineno, body[0].end_lineno + 1):
                    found[ln] = lines[ln - 1].strip()
    return found


def main(argv: list[str]) -> int:
    strict = "--strict" in argv
    flagged = 0
    heavy: list[str] = []
    files = sorted(p for p in TREE.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        source = path.read_text(encoding="utf8")
        total = sum(1 for line in source.splitlines() if line.strip())
        found = comment_lines(source)
        for ln, text in sorted(found.items()):
            if MEASURE.search(text) or HISTORY.search(text):
                print(f"{path.relative_to(ROOT)}:{ln}: measurement or history in a comment: {text[:90]}")
                flagged += 1
        if total >= 40 and len(found) / total > SHARE_LIMIT:
            heavy.append(f"{path.relative_to(ROOT)}: {round(100 * len(found) / total)}% comments/docstrings")
    for line in heavy:
        print(line)
    print(f"check_comments: {len(files)} files, {flagged} flagged line(s), {len(heavy)} comment-heavy file(s)")
    return 1 if strict and (flagged or heavy) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
