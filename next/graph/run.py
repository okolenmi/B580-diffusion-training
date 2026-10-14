"""Headless runner: validate and run a graph file without a server.

    python -m next.graph.run graph.json [--validate-only] [--package PKG ...]
"""

from __future__ import annotations

import argparse
import json
import sys

from .document import GraphFormatError, parse
from .registry import NodeRegistry
from .runtime import GraphRunner


class _PrintReporter:
    def __init__(self, quiet: bool):
        self.quiet = quiet

    def report(self, stream: str, frame: dict) -> None:
        if not self.quiet:
            print(f"[{stream}] {json.dumps(frame, default=str)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m next.graph.run")
    parser.add_argument("graph")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--package", action="append", help="node package to scan (default: the tree's nodes)")
    args = parser.parse_args(argv)
    try:
        with open(args.graph, encoding="utf8") as handle:
            doc = parse(json.load(handle))
    except (OSError, ValueError, GraphFormatError) as exc:
        print(f"cannot read graph: {exc}", file=sys.stderr)
        return 2
    registry = NodeRegistry(packages=tuple(args.package)) if args.package else NodeRegistry.default()
    runner = GraphRunner(registry, reporter=_PrintReporter(args.quiet))
    validation = runner.validate(doc)
    for issue in validation.issues:
        print(f"{issue.severity}: {issue.code}: {issue.message}")
    if args.validate_only or not validation.ok:
        return 0 if validation.ok else 1
    outcome = runner.run(doc)
    print("run:", "ok" if outcome.ok else f"failed: {outcome.error}", "(stopped)" if outcome.stopped else "")
    return 0 if outcome.ok else 1


if __name__ == "__main__":
    sys.exit(main())
