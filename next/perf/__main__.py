"""``ceilings [--device xpu]`` measures what the card can do; ``report FOLDER`` re-renders a saved report.json (findings included)
after the rules changed; ``bench`` and ``hardware`` and ``allocator`` measure one thing each; ``hwcheck -- TRAIN ARGS`` runs them all with the default scenarios; ``matrix FILE`` runs scenarios; ``compare A B`` prints what changed between two report folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ceilings import load_or_measure
from .findings import findings_for
from .comparison import Comparison
from .bench import SUITES, BenchRunner, context_for
from .hardware import AllocatorProbe, HardwareReport
from .hwcheck import HwCheck
from .matrix import Matrix, MatrixSpec, SubprocessRunner
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
    matrix = sub.add_parser("matrix", help="run every scenario of a TOML file in its own process and write one summary")
    matrix.add_argument("file")
    matrix.add_argument("--out", default="perf-matrix")
    matrix.add_argument("--resume", action="store_true", help="keep scenarios whose report.json already exists")
    compare = sub.add_parser("compare", help="what changed between two report folders")
    compare.add_argument("baseline")
    compare.add_argument("other")
    hardware = sub.add_parser("hardware", help="versions, device and environment as hardware.md and hardware.json")
    hardware.add_argument("--out", default="perf-hardware")
    allocator = sub.add_parser("allocator", help="print what the allocator reserves in a fragmenting pattern (run under different PYTORCH_ALLOC_CONF)")
    allocator.add_argument("--device", default="xpu")
    allocator.add_argument("--quick", action="store_true")
    bench = sub.add_parser("bench", help="micro-benchmarks of one operation at a time, rated against the ceilings")
    bench.add_argument("--device", default="xpu")
    bench.add_argument("--quick", action="store_true", help="tiny sizes (a CPU, a smoke run)")
    bench.add_argument("--out", default="perf-bench")
    bench.add_argument("--batch", type=int, default=2)
    bench.add_argument("--size", type=int, default=768, help="image side in pixels")
    bench.add_argument("--suites", nargs="*", choices=sorted(SUITES), help="default: all")
    bench.add_argument("--backend", default="auto", choices=("auto", "xpu_kernels", "torch"))
    check = sub.add_parser("hwcheck", help="everything pending on hardware in one go; train arguments follow --")
    check.add_argument("--out", default="perf-hwcheck")
    check.add_argument("--device", default="xpu")
    check.add_argument("--steps", type=int, default=28)
    check.add_argument("--warmup", type=int, default=20, help="steps before the profiled window (graph capture and timing need several sightings per shape)")
    check.add_argument("--only", nargs="*", choices=("hardware", "allocator", "bench", "matrix", "verdict"), help="default: all")
    check.add_argument("train_args", nargs=argparse.REMAINDER, help="after --: --models M --datasets D --file F --dataset NAME --output OUT, plus --batch/--resolution/...")
    args = parser.parse_args(argv)
    if args.command == "hardware":
        info = HardwareReport()
        data = info.collect()
        folder = Path(args.out)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "hardware.json").write_text(json.dumps(data, indent=1))
        (folder / "hardware.md").write_text(info.markdown(data))
        print(folder / "hardware.md")
        return 0
    if args.command == "allocator":
        print(json.dumps(AllocatorProbe(args.device, args.quick).run()))
        return 0
    if args.command == "bench":
        ctx = context_for(args.device, args.quick, args.out, args.batch, args.size, args.backend)
        runner = BenchRunner(ctx, args.suites)
        print(runner.write(runner.run(), args.out))
        return 0
    if args.command == "hwcheck":
        train = [a for a in args.train_args if a != "--"]
        if not train:
            raise SystemExit("give the training arguments after --, e.g.: hwcheck -- --models M --datasets D --file F --dataset NAME --output OUT")
        print(HwCheck(Path(args.out), train, device=args.device, steps=args.steps, warmup=args.warmup, only=set(args.only) if args.only else None).run())
        return 0
    if args.command == "matrix":
        spec = MatrixSpec.load(args.file)
        print(Matrix(spec, Path(args.out), SubprocessRunner(spec.command, spec.timeout_s), resume=args.resume).run())
        return 0
    if args.command == "compare":
        load = lambda folder: json.loads((Path(folder) / "report.json").read_text())  # noqa: E731
        print(Comparison(load(args.baseline), load(args.other), args.baseline, args.other).markdown())
        return 0
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
