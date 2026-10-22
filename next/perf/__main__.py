"""``python -m next.perf ceilings [--device xpu]`` measures what the card can do; ``report FOLDER`` re-renders a saved report.json
(findings included) after the rules changed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ceilings import load_or_measure
from .findings import findings_for
from .report import ReportWriter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m next.perf")
    sub = parser.add_subparsers(dest="command", required=True)
    ceilings = sub.add_parser("ceilings", help="measure matmul, int8, copy and launch ceilings and print them")
    ceilings.add_argument("--device", default="xpu")
    ceilings.add_argument("--quick", action="store_true", help="small sizes (a CPU, a smoke run)")
    ceilings.add_argument("--cache", default="ceilings.json")
    report = sub.add_parser("report", help="re-render report.md from FOLDER/report.json with the current findings rules")
    report.add_argument("folder")
    args = parser.parse_args(argv)
    if args.command == "ceilings":
        print(json.dumps(load_or_measure(args.cache, args.device, args.quick).to_json(), indent=1))
        return 0
    folder = Path(args.folder)
    data = json.loads((folder / "report.json").read_text())
    data["findings"] = [vars(f) for f in findings_for(data)]
    print(ReportWriter(folder).write(data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
