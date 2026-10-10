#!/usr/bin/env python3
"""Repo-relative path references in backticks: do they exist? (E3)

Scans tracked *.md/*.py/*.toml/*.sh for `like/this/path.py` spans and
reports the ones that resolve to nothing under the repo root. The two
archive moves on 2026-10-09 (bad-LoRA-investigation, shapes stall) are
exactly the kind of change that orphans these; nothing renders the docs
and check_doc_links.py only resolves markdown links/anchors, so without
this a citation rots silently.

Only git-tracked files are scanned (runs/, datasets/, caches are
gitignoredcratch, not references). Exit 1 with the list when anything
is orphaned, else exit 0. Keep it fast and false-positive-free: a
finding here must mean a real dead reference, never a heuristic
misfire -- patterns that misfire get narrowed, not the files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Backtick spans that are URLs, shell, key names or prose -- never paths.
_SKIP = re.compile(
    r"^(https?://|www\.|/|[~$]|.*\s.*--|.*\|\||.*&&|.*\$\(|\.\.\.|--)"
    r"|.*://.*|.*\+.*\/.*|.*[\{\}<>].*|.*\*\*.*|.*\*.*"
)


def _resolve(root: Path, base: str) -> bool:
    if (root / base).exists():
        return True
    # Extensionless module refs (`backend/__init__`): try +".py".
    if "." not in base.rsplit("/", 1)[-1]:
        return (root / (base + ".py")).exists()
    return False

#: What a repo-relative path looks like: has a slash, no spaces, no URL
#: scheme, ends in a plausible tail (extension, or a known directory).
_EXT = re.compile(r"\.(py|md|toml|sh|json|jsonl|safetensors|csv|png|txt|yaml|yml|db|js|css|html)$")


#: First path segments that anchor a span to the repo root. Only these
#: are checked: backend source comments use module-relative shorthands
#: (`application/limits.py`) that are not repo paths, and flagging those
#: would drown the real orphans. Current top-level entries plus retired
#: names (so stale references to moved things still get caught).
_ROOTS = frozenset({
    "nodes", "backend", "docs", "scripts", "manager",
    "archive", "datasets", "paths.py", "run_tests.py",
    "run_server.sh", "monitor_bus.py", "config.example.toml",
    "convert-cfg.toml", "requirements.txt", "requirements-dev.txt",
    "pyproject.toml", "README.md",
    # retired: referencing these means pointing at a move
    "core", "server", "bad-LoRA-investigation",
    "shapes diversity problem", "shapes diversity problem 2",
})
# frontend/ is unchecked: design docs cite planned UI files that were
# never built (views/run.js), and flagging those would rewrite history.

#: Bare retired-dir names (no file part) are not checked: frozen
#: comparative prose, and the live architecture claim is a maintainer
#: matter, not a file reference.
_UNCHECKED_BARE = frozenset({"core/", "server/"})


def _candidates(text: str) -> list[str]:
    out = []
    for m in re.finditer(r"`([^`\n]+)`", text):
        span = m.group(1).strip()
        if "/" not in span or " " in span or _SKIP.match(span):
            continue
        # Trailing punctuation from prose ("see foo/bar.py,") is not
        # part of the path.
        span = span.rstrip(".,:;)]")
        # Line-number suffixes ("file.py:123") point at the file.
        base = span.split(":")[0] if _EXT.search(span.split(":")[0]) else span
        first, _, _ = base.partition("/")
        if first not in _ROOTS:
            continue
        if base in _UNCHECKED_BARE:
            continue
        out.append(base)
    return out


def main() -> int:
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files",
         "*.md", "*.py", "*.toml", "*.sh"],
        capture_output=True, text=True, check=True).stdout.splitlines()
    # archive/ is frozen history: its references were correct at write time
    # and "fixing" them would falsify the record. Live tree only.
    tracked = [r for r in tracked if not r.startswith("archive/")]
    bad: dict[str, list[str]] = {}
    for rel in tracked:
        path = REPO / rel
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for cand in _candidates(text):
            if not _resolve(REPO, cand):
                bad.setdefault(rel, []).append(cand)
    if bad:
        print("dead repo-relative references in backticks:")
        for rel in sorted(bad):
            seen = sorted(set(bad[rel]))
            print(f"  {rel}: {', '.join(seen)}")
        return 1
    print(f"doc refs: {len(tracked)} files scanned, all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
