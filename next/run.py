"""Run the new trainer from the command line, with the real model library and dataset store.

    python -m next.run ingest --datasets D:/datasets my_set D:/images
    python -m next.run sdxl-lora --models D:/models --datasets D:/datasets --file sdxl_base.safetensors \\
        --dataset my_set --output D:/out --resolution 1024 --batch 2 --rank 16 --steps 200
    python -m next.run graph graph.json --models D:/models --datasets D:/datasets

``--file`` is relative to ``<models>/checkpoints`` when that folder exists, otherwise to ``--models`` itself.
``--set key=value`` overrides a graph setting (``--set memory.vram_max_mb=9000``) or a node's parameter in the starter graph (``--set adapter.attention_fraction=0.5``); ``--param`` is the same for parameters only. Memory is printed as
the breakdown the residency manager reports; the lines are the monitor's data. Every finished node prints its duration
and the card's memory; ``--memory-log run.csv`` also samples memory four times a second against what the run is doing.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from dataclasses import replace
from pathlib import Path

from . import starters
from .console import Console, NodeTrace
from .data.store import DatasetStore
from .engine.devices import Accelerator, resolve_device
from .engine.xpu_env import set_perf_env
from .graph.settings import SettingsRegistry
from .engine.runtime.factory import make_runtime
from .engine.runtime.sampler import MemorySampler
from .families.api.library import ModelLibrary
from .graph import GraphRunner, NodeRegistry, parse
from .graph.node import Environment


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


def _split_settings(pairs: list[str] | None) -> tuple[list[str], list[str]]:
    """``--set`` takes both kinds: a registered graph setting (``memory.vram_max_mb``) is one, anything else is read as
    ``node.port``."""
    known = SettingsRegistry()
    settings, params = [], []
    for pair in pairs or []:
        (settings if known.spec(pair.partition("=")[0]) is not None else params).append(pair)
    return settings, params


def _overrides(pairs: list[str] | None) -> dict[str, dict]:
    """``--param node.port=value`` pairs as ``{node: {port: value}}``; values are JSON when they parse, text otherwise."""
    out: dict[str, dict] = {}
    for pair in pairs or []:
        target, _, raw = pair.partition("=")
        node, _, port = target.partition(".")
        if not node or not port:
            raise SystemExit(f"--param wants node.port=value, got {pair!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        out.setdefault(node, {})[port] = value
    return out


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
    if getattr(args, "perf", None):
        doc = replace(doc, settings={**doc.settings, "perf.dir": args.perf})
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


def _starter(args, settings: list[str], params: list[str]):
    return starters.sdxl_lora(model=args.file, dataset=args.dataset, output=args.output, name=args.name, resolution=args.resolution,
        batch_size=args.batch, accumulation=args.accumulation, steps=args.steps if args.steps is not None else (0 if args.epochs else 1000), epochs=args.epochs, save_every_epoch=args.save_every_epoch, auto_caption=args.auto_caption, lr=args.lr, rank=args.rank,
        optimizer=args.optimizer, fused=args.fused, save_every=args.save_every,
        checkpointing=not args.no_checkpointing, persist_conditioning=args.persist_conditioning, max_shapes=args.max_shapes, batch_area=args.batch_area, pack=args.pack, fill=args.pad_fill,
        time_balance=args.time_balance, balance_sampling=args.balance_sampling, graph=args.graph, schedule=args.schedule, tag_swap=args.tag_swap, save_probe=args.save_probe,
        settings=_settings(settings), overrides=_overrides([*_regions_preset(args.regions), *params, *(args.param or [])]))


def _regions_preset(regions: int) -> list[str]:
    """``--regions N``: every image at N timesteps spread over the range, each region's gradient scaled to the same length (settings given later win)."""
    return [f"noiser.regions={regions}", "noiser.layout=strata", "trainer.gradient_policy=equal"] if regions > 1 else []


def _parser() -> argparse.ArgumentParser:
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
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="a graph setting (memory.vram_max_mb=9000) or a node parameter of the starter graph (trainer.graph=true)")
        p.add_argument("--param", action="append", metavar="NODE.PORT=VALUE",
                       help="a node's parameter by the node's name in the starter (for example adapter.attention_fraction=0.5)")
        p.add_argument("--verbose", action="store_true")
        p.add_argument("--perf", nargs="?", const="perf-report", metavar="DIR",
                       help="profile a few steps of the trainer (after a warm-up) and write a rated performance report to DIR (default perf-report)")
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
            p.add_argument("--steps", type=int, default=None, help="optimizer steps (default 1000, or no limit when --epochs is given)")
            p.add_argument("--epochs", type=int, default=0, help="passes over the data; with --steps the run ends at whichever limit comes first")
            p.add_argument("--auto-caption", action="store_true", help="caption images that have no caption first (settings: --param autocaption.model_id=...)")
            p.add_argument("--save-every-epoch", type=int, default=0, help="also save after every this many epochs (0 = off)")
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--schedule", default="constant", choices=("constant", "cosine"), help="learning rate over the run: constant, or a cosine decay to 5%% of --lr over the whole run")
            p.add_argument("--save-probe", action="store_true", help="before each save, run one forward with the live adapter and keep it in <file>.probe.safetensors "
                           "(python -m next.accept save-probe checks the saved file against it)")
            p.add_argument("--regions", type=int, default=1, help="timesteps per image in one step, one from each equal part of the range, gradients of the parts "
                           "scaled to equal length before they are averaged (N times the batch per step; see manual/quality.md, 'Timesteps per image')")
            p.add_argument("--tag-swap", action="store_true", help="train each epoch on one random tag of every caption instead of the whole caption (an experiment)")
            p.add_argument("--rank", type=int, default=16)
            p.add_argument("--optimizer", default="AdamWNode", choices=("AdamWNode", "AdafactorNode", "CAMENode"))
            p.add_argument("--fused", action="store_true")
            p.add_argument("--save-every", type=int, default=250)
            p.add_argument("--no-checkpointing", action="store_true")
            p.add_argument("--persist-conditioning", action="store_true",
                           help="keep text encodings of stable prompts on disk next to the dataset's latents")
            p.add_argument("--max-shapes", type=int, default=16,
                           help="most distinct padded shapes the model will see (0 = no limit); each new shape costs seconds and driver memory")
            p.add_argument("--batch-area", type=float, default=0.0,
                           help="a batch holds at most batch size times this many base-resolution areas in total (e.g. 1.5), so bigger samples run in smaller batches; 0 = no limit")
            p.add_argument("--pad-fill", default="zero", choices=("zero", "mirror"),
                           help="what the padded part of the network input holds: a blank strip, or a reflection of the sample's own noisy input")
            p.add_argument("--pack", type=float, default=0.3,
                           help="fill short batches by moving left-over samples into larger shapes when no sample is more than this share padding; 0 = off")
            p.add_argument("--graph", action="store_true", help="replay forward and backward from captured device graphs (XPU; needs no dropout, no fused optimizer)")
            p.add_argument("--time-balance", default="off", choices=("off", "normalize", "speed", "dro", "gradnorm"),
                           help="weight the loss by how each third of the timestep range is improving; gradnorm: by the inverse of the typical gradient norm of the third (batch 1, one timestep per step)")
            p.add_argument("--balance-sampling", type=float, default=0.0,
                           help="draw timesteps more often from the thirds that are behind (power of loss/baseline; needs tracking, 1 is a start)")
            p.add_argument("--dump", help="write the graph here instead of running it")
    return parser


def _ingest(args) -> int:
    library = DatasetStore(args.datasets).library(args.name, create=True)
    result = library.ingest(args.folder, progress=lambda done, total: print(f"\r{done}/{total}", end="", flush=True))
    print(f"\nadded {result.added}, already known {result.known}, skipped {len(result.skipped)}")
    for path, reason in result.skipped[:10]:
        print(f"  skipped {path}: {reason}")
    library.close()
    return 0


def _graph_file(args):
    """A graph file as written; ``--set`` here can only be a graph setting, since node parameters live in the file."""
    with open(args.graph, encoding="utf8") as handle:
        doc = parse(json.load(handle))
    settings, params = _split_settings(args.set)
    if params or args.param:
        raise SystemExit(f"a graph file sets node parameters inside the file; --set takes graph settings here (not {(params + (args.param or []))[0]!r})")
    return replace(doc, settings={**doc.settings, **_settings(settings)})


def main(argv: list[str] | None = None) -> int:
    set_perf_env()
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "graph":
        return _run(_graph_file(args), args)
    settings, params = _split_settings(args.set)
    try:
        doc = _starter(args, settings, params)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.dump:
        Path(args.dump).write_text(json.dumps(doc.as_dict(), indent=2), encoding="utf8")
        return 0
    return _run(doc, args)


if __name__ == "__main__":
    sys.exit(main())
