"""Check that documentation links and code citations resolve.

Two failure modes this catches, both of which had already happened in
this repo before the 2026-10-01 cleanup:

* a markdown link pointing at a file or heading that does not exist --
  these rot silently because nothing ever renders the docs;
* a source comment or docstring citing ``docs/some/file.md`` that was
  deleted or renamed, so the comment now points at nothing (there were
  two: a long-deleted ``docs/CLEANUP_TODO.md`` and an abbreviated
  ``09-...md`` path).

Run:  python scripts/check_doc_links.py [--quiet]
Gate: part of scripts/full_gate.sh.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Source trees whose comments and docstrings cite docs.
CODE_GLOBS = ("nodes/**/*.py", "core/**/*.py", "manager/**/*.py", "backend/**/*.py",
              "frontend/**/*.js", "scripts/**/*.py", "scripts/**/*.sh",
              "*.py", "*.sh")

#: ``docs/....md`` in running prose or a comment (optionally with #anchor).
CITATION = re.compile(r"\bdocs/[A-Za-z0-9_./-]*?\.md(?:#[A-Za-z0-9_.-]+)?")

#: [text](target) links, plus bare images.
MD_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")

#: Heading text -> slug, the way GitHub builds anchors.
_SLUG_STRIP = re.compile(r"[^\w\s-]")
_SLUG_SPACE = re.compile(r"\s+")

#: This file quotes example doc paths in its own prose; it must not
#: check itself.
SELF = Path(__file__).name


def slugify(heading: str) -> str:
    text = heading.strip().lower()
    text = text.strip("`")
    text = _SLUG_STRIP.sub("", text)
    return _SLUG_SPACE.sub("-", text)


def headings_of(path: Path) -> set[str]:
    """Every anchor a file offers: its own slugged headings, plus the
    implicit ``#`` top-of-file target."""
    anchors = {""}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                anchors.add(slugify(line.lstrip("#")))
    except OSError:
        pass
    return anchors


def iter_source_files() -> list[Path]:
    seen: set[Path] = set()
    for pattern in CODE_GLOBS:
        for path in REPO.glob(pattern):
            if "__pycache__" in path.parts or "node_modules" in path.parts:
                continue
            seen.add(path)
    return sorted(seen)


def check_citations(problems: list[str], verbose: bool) -> int:
    """Every ``docs/**.md`` mentioned in source must exist."""
    checked = 0
    for path in iter_source_files():
        if path.name == SELF:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in CITATION.finditer(text):
            target, _, anchor = match.group(0).partition("#")
            checked += 1
            doc = REPO / target
            if not doc.is_file():
                problems.append(
                    f"{path.relative_to(REPO)}: cites missing doc {target}"
                )
            elif anchor and anchor not in headings_of(doc):
                problems.append(
                    f"{path.relative_to(REPO)}: cites {target}#{anchor}, "
                    "no such heading"
                )
    if verbose:
        print(f"  {checked} source citation(s) checked")
    return checked


def check_markdown_links(problems: list[str], verbose: bool) -> int:
    """Every relative markdown link must resolve to a real file."""
    checked = 0
    for doc in sorted(REPO.glob("docs/**/*.md")) + [REPO / "README.md"]:
        try:
            text = doc.read_text(encoding="utf-8")
        except OSError:
            continue
        for match in MD_LINK.finditer(text):
            target = match.group(1).strip().split(" ")[0]
            if not target or target.startswith(("http://", "https://", "mailto:")):
                continue
            if target.startswith("#"):
                if target[1:] not in headings_of(doc):
                    problems.append(f"{doc.relative_to(REPO)}: dead anchor {target}")
                continue
            checked += 1
            path_part, _, anchor = target.partition("#")
            if path_part:
                resolved = (doc.parent / path_part).resolve()
                if not resolved.exists():
                    problems.append(
                        f"{doc.relative_to(REPO)}: link to missing {path_part}"
                    )
                    continue
                if resolved.is_dir():
                    index = resolved / "README.md"
                    if not index.is_file():
                        problems.append(
                            f"{doc.relative_to(REPO)}: directory link without README: "
                            f"{path_part}"
                        )
                    continue
            if anchor and resolved.suffix in {".md", ""} and anchor not in headings_of(resolved):
                problems.append(
                    f"{doc.relative_to(REPO)}: {target} has no heading '{anchor}'"
                )
    if verbose:
        print(f"  {checked} markdown link(s) checked")
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print("== documentation links ==")
    problems: list[str] = []
    check_markdown_links(problems, verbose)
    if verbose:
        print("== doc citations from source ==")
    check_citations(problems, verbose)

    if problems:
        print(f"\nDOC LINKS: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    if verbose:
        print("DOC LINKS: all resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())