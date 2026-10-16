"""Run the new trainer from the command line, with the real model library and dataset store.

    python -m next.run ingest --datasets D:/datasets my_set D:/images
    python -m next.run sdxl-lora --models D:/models --datasets D:/datasets --file checkpoints/sdxl_base.safetensors \\
        --dataset my_set --output D:/out --resolution 1024 --batch 2 --rank 16 --steps 200
    python -m next.run graph graph.json --models D:/models --datasets D:/datasets

``--set key=value`` overrides a graph setting (for example ``--set memory.vram_max_mb=9000``). Memory is printed as
the breakdown the residency manager reports; the lines are the monitor's data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import starters
from .data.store import DatasetStore
from .engine.runtime.factory import make_runtime
from .families.api.library import ModelLibrary
from .graph import GraphRunner, NodeRegistry, parse
from .graph.node import Environment


class Console:
    """Prints step metrics and memory frames; the rest is quiet unless ``verbose``."""

    def __init__(self, verbose: bool):
        self.verbose = verbose

    def report(self, stream: str, frame: dict) -> None:
        if stream == "memory":
            self._memory(frame)
        elif frame.get("name") == "step_finished":
            self._step(frame["summary"])
        elif stream == "warning" or self.verbose:
            print(f"[{stream}] {json.dumps(frame, default=str)}", flush=True)

    @staticmethod
    def _step(summary: dict) -> None:
        m = summary["metrics"]
        memory = f"  device {m['device_mb']:.0f} MB (live {m['live_mb']:.0f}, step peak {m['step_peak_mb']:.0f}, cached {m['reserved_mb']:.0f})" \
            if "device_mb" in m else ""
        print(f"step {summary['step']}/{summary['total']}  loss {m['loss']:.4f}  {m['step_ms']:.0f} ms{memory}", flush=True)

    @staticmethod
    def _memory(frame: dict) -> None:
        if frame["event"] == "breakdown":
            parts = "  ".join(f"{name} {mb:.0f}" for name, mb in frame["parts"].items())
            print(f"memory (MB) at step {frame['step']}: {parts}", flush=True)
        else:
            print(f"memory: {json.dumps(frame, default=str)}", flush=True)


def _environment(args) -> Environment:
    services = {}
    if getattr(args, "datasets", None):
        services["datasets"] = DatasetStore(args.datasets)
    if getattr(args, "models", None):
        root = Path(args.models)
        folders = {scope: [root / scope] for scope in ("checkpoints", "vae", "loras", "text_encoders") if (root / scope).is_dir()}
        services["models"] = ModelLibrary(folders or {"checkpoints": [root]})
    return Environment(services=services)


def _settings(pairs: list[str] | None) -> dict:
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            out[key] = json.loads(raw)
        except ValueError:
            out[key] = raw
    return out


def _run(doc, args) -> int:
    runner = GraphRunner(NodeRegistry.default(), reporter=Console(args.verbose), env=_environment(args), runtime_factory=make_runtime)
    validation = runner.validate(doc)
    for issue in validation.issues:
        print(f"{issue.severity}: {issue.code}: {issue.message}")
    if not validation.ok:
        return 1
    outcome = runner.run(doc)
    print("run:", "ok" if outcome.ok else f"failed: {outcome.error}", "(stopped)" if outcome.stopped else "")
    return 0 if outcome.ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m next.run")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="register a folder of images (and sidecar captions) as a dataset")
    ingest.add_argument("--datasets", required=True)
    ingest.add_argument("name")
    ingest.add_argument("folder")
    for name in ("sdxl-lora", "graph"):
        p = sub.add_parser(name)
        p.add_argument("--models", required=True)
        p.add_argument("--datasets", required=True)
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="graph setting")
        p.add_argument("--verbose", action="store_true")
        if name == "graph":
            p.add_argument("graph")
        else:
            p.add_argument("--file", required=True, help="checkpoint relative to <models>/checkpoints")
            p.add_argument("--dataset", required=True)
            p.add_argument("--output", required=True)
            p.add_argument("--name", default="lora")
            p.add_argument("--resolution", type=int, default=1024)
            p.add_argument("--batch", type=int, default=1)
            p.add_argument("--accumulation", type=int, default=1)
            p.add_argument("--steps", type=int, default=1000)
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--rank", type=int, default=16)
            p.add_argument("--optimizer", default="AdamWNode", choices=("AdamWNode", "AdafactorNode", "CAMENode"))
            p.add_argument("--fused", action="store_true")
            p.add_argument("--save-every", type=int, default=250)
            p.add_argument("--no-checkpointing", action="store_true")
            p.add_argument("--dump", help="write the graph here instead of running it")
    args = parser.parse_args(argv)
    if args.command == "ingest":
        library = DatasetStore(args.datasets).library(args.name, create=True)
        result = library.ingest(args.folder, progress=lambda done, total: print(f"\r{done}/{total}", end="", flush=True))
        print(f"\nadded {result.added}, already known {result.known}, skipped {len(result.skipped)}")
        for path, reason in result.skipped[:10]:
            print(f"  skipped {path}: {reason}")
        library.close()
        return 0
    if args.command == "graph":
        with open(args.graph, encoding="utf8") as handle:
            doc = parse(json.load(handle))
    else:
        doc = starters.sdxl_lora(model=args.file, dataset=args.dataset, output=args.output, name=args.name, resolution=args.resolution,
                                 batch_size=args.batch, accumulation=args.accumulation, steps=args.steps, lr=args.lr, rank=args.rank,
                                 optimizer=args.optimizer, fused=args.fused, save_every=args.save_every,
                                 checkpointing=not args.no_checkpointing, settings=_settings(args.set))
        if args.dump:
            Path(args.dump).write_text(json.dumps(doc.as_dict(), indent=2), encoding="utf8")
            return 0
    if args.command == "graph":
        doc = replace(doc, settings={**doc.settings, **_settings(args.set)})
    return _run(doc, args)


if __name__ == "__main__":
    sys.exit(main())
