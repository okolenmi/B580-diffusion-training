"""Run the new trainer from the command line, with the real model library and dataset store.

    python -m next.run ingest --datasets D:/datasets my_set D:/images
    python -m next.run sdxl-lora --models D:/models --datasets D:/datasets --file sdxl_base.safetensors \\
        --dataset my_set --output D:/out --resolution 1024 --batch 2 --rank 16 --steps 200
    python -m next.run graph graph.json --models D:/models --datasets D:/datasets

``--file`` is relative to ``<models>/checkpoints`` when that folder exists, otherwise to ``--models`` itself.
``--set key=value`` overrides a graph setting (for example ``--set memory.vram_max_mb=9000``). Memory is printed as
the breakdown the residency manager reports; the lines are the monitor's data. Every finished node prints its duration
and the card's memory; ``--memory-log run.csv`` also samples memory four times a second against what the run is doing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import starters
from .data.store import DatasetStore
from .engine.devices import Accelerator, resolve_device
from .engine.xpu_env import set_perf_env
from .engine.runtime.factory import make_runtime
from .engine.runtime.sampler import MemorySampler
from .families.api.library import ModelLibrary
from .graph import GraphRunner, NodeRegistry, parse
from .graph.node import Environment


class Console:
    """Prints step metrics and memory frames; the rest is quiet unless ``verbose``."""

    def __init__(self, verbose: bool):
        self.verbose = verbose
        self.sampler: MemorySampler | None = None

    def report(self, stream: str, frame: dict) -> None:
        if stream == "memory":
            self._memory(frame)
        elif frame.get("name") == "step_finished":
            self._step(frame["summary"])
            if self.sampler is not None:
                self.sampler.label = f"step {frame['summary']['step'] + 1}"
        elif stream == "buckets":
            print(f"data: {frame['samples']} samples, {frame['shapes_in']} latent shapes -> {frame['bucket_shapes']} padded shapes "
                  f"({frame['batch_shapes']} with short batches) in {frame['batches']} batches; padding mean {frame['padding_mean']:.0%}, "
                  f"worst {frame['padding_worst']:.0%}", flush=True)
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
            retries = "" if frame.get("alloc_retries") is None else f", allocator cache flushes {frame['alloc_retries']:.0f} in {frame['segments']:.0f} segments"
            print(f"memory (MB) at step {frame['step']}: needs {frame['need_mb']}, peak live {frame['peak_mb']}, card {frame['used_mb']}{retries}", flush=True)
            for name, mb in frame["parts"].items():
                print(f"  {mb:8.0f}  {name}", flush=True)
        elif frame["event"] == "claims":
            print("memory claims (MB):", flush=True)
            for row in frame["rows"]:
                print(f"  {row['mb']:8.0f}  {row['name']:<22} {row['purpose']}", flush=True)
        else:
            print(f"memory: {json.dumps(frame, default=str)}", flush=True)


def _environment(args) -> Environment:
    services = {}
    if getattr(args, "datasets", None):
        services["datasets"] = DatasetStore(args.datasets)
    if getattr(args, "models", None):
        root = Path(args.models)
        folders = {scope: [root / scope] for scope in ("vae", "loras", "text_encoders") if (root / scope).is_dir()}
        folders["checkpoints"] = [root / "checkpoints" if (root / "checkpoints").is_dir() else root]
        services["models"] = ModelLibrary(folders)
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


class NodeTrace:
    """One line per finished node: how long it took and where the card's memory stood, plus the sampler's label."""

    def __init__(self, doc, accelerator: Accelerator, sampler: MemorySampler | None):
        self.names = {spec.uid: spec.class_name for spec in doc.nodes}
        self.accelerator, self.sampler = accelerator, sampler
        self.t0 = time.monotonic()

    def node_state(self, uid: str, state: str, detail: dict) -> None:
        if self.sampler is not None:
            self.sampler.label = f"{self.names.get(uid, uid)} {state}"

    def node_done(self, result) -> None:
        stats = self.accelerator.memory()
        memory = (f"  allocated {stats['allocated']:.0f}, reserved {stats['reserved']:.0f}, device {stats['device_used']:.0f} MB"
                  if "device_used" in stats and "allocated" in stats and "reserved" in stats else "")
        print(f"[{time.monotonic() - self.t0:6.1f}s] {self.names.get(result.uid, result.uid)}: {result.duration_ms / 1000:.1f}s{memory}", flush=True)

    def run(self, state: str, error: str | None) -> None:
        pass


def _run(doc, args) -> int:
    device = resolve_device(doc.settings.get("device.default", "auto"))
    accelerator = Accelerator(device)
    console = Console(args.verbose)
    sampler = MemorySampler(accelerator, args.memory_log) if args.memory_log else None
    trace = NodeTrace(doc, accelerator, sampler)
    runner = GraphRunner(NodeRegistry.default(), reporter=console, env=_environment(args), runtime_factory=make_runtime, observer=trace)
    validation = runner.validate(doc)
    for issue in validation.issues:
        print(f"{issue.severity}: {issue.code}: {issue.message}")
    if not validation.ok:
        return 1
    console.sampler = sampler
    with (sampler if sampler is not None else contextlib.nullcontext()):
        outcome = runner.run(doc)
    print("run:", "ok" if outcome.ok else f"failed: {outcome.error}", "(stopped)" if outcome.stopped else "")
    return 0 if outcome.ok else 1


def main(argv: list[str] | None = None) -> int:
    set_perf_env()
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
        p.add_argument("--memory-log", metavar="FILE.csv", help="sample device memory four times a second, labelled by node and step")
        if name == "graph":
            p.add_argument("graph")
        else:
            p.add_argument("--file", required=True, help="checkpoint file, relative to the checkpoints folder (or to --models when it has no checkpoints subfolder)")
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
            p.add_argument("--persist-conditioning", action="store_true",
                           help="keep text encodings of stable prompts on disk next to the dataset's latents")
            p.add_argument("--max-shapes", type=int, default=16,
                           help="most distinct padded shapes the model will see (0 = no limit); each new shape costs seconds and driver memory")
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
                                 checkpointing=not args.no_checkpointing, persist_conditioning=args.persist_conditioning, max_shapes=args.max_shapes,
                                 settings=_settings(args.set))
        if args.dump:
            Path(args.dump).write_text(json.dumps(doc.as_dict(), indent=2), encoding="utf8")
            return 0
    if args.command == "graph":
        doc = replace(doc, settings={**doc.settings, **_settings(args.set)})
    return _run(doc, args)


if __name__ == "__main__":
    sys.exit(main())
