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
from dataclasses import fields, replace
from pathlib import Path

from . import starters
from .starters import SdxlLoraOptions
from .console import Console, NodeTrace
from .data.store import DatasetStore
from .engine.devices import Accelerator, resolve_device
from .engine.xpu_env import set_perf_env
from .graph.settings import SettingsRegistry
from .engine.runtime.factory import make_runtime
from .engine.runtime.sampler import MemorySampler
from .families.api.library import ModelLibrary
from .graph import AddonError, GraphRunner, NodeRegistry, load_set, parse
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


DEFAULTS = SdxlLoraOptions()


def _starter(args, settings: list[str], params: list[str]):
    """The starter graph from the command line: every argument whose destination is a field of ``SdxlLoraOptions`` is passed on."""
    options = {k: v for k, v in vars(args).items() if k in {f.name for f in fields(SdxlLoraOptions)}}
    if options.get("gradient_policy") is None:
        options["gradient_policy"] = "equal" if args.regions > 1 else "mean"
    options["stages"] = tuple(args.stages or ())
    options["steps"] = args.steps if args.steps is not None else (0 if args.epochs else DEFAULTS.steps)
    return starters.sdxl_lora(model=args.file, dataset=args.dataset, output=args.output, **options,
                              settings=_settings(settings), overrides=_overrides([*_regions_preset(args.regions), *params, *(args.param or [])]))


def _regions_preset(regions: int) -> list[str]:
    """``--regions N``: every image at N timesteps spread over the range, each region's gradient scaled to the same length (settings given later win)."""
    return [f"noiser.regions={regions}", "noiser.layout=strata"] if regions > 1 else []


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
        p.add_argument("--addons", metavar="FILE.json",
                       help="an addon set: JSON mapping a node (title, class or uid) or 'graph' to addons to switch on (rework/31-node-addons.md)")
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
            p.add_argument("--name", default=DEFAULTS.name)
            p.add_argument("--resolution", type=int, default=DEFAULTS.resolution)
            p.add_argument("--batch", dest="batch_size", type=int, default=DEFAULTS.batch_size)
            p.add_argument("--accumulation", type=int, default=DEFAULTS.accumulation)
            p.add_argument("--steps", type=int, default=None, help="optimizer steps (default 1000, or no limit when --epochs is given)")
            p.add_argument("--epochs", type=int, default=DEFAULTS.epochs, help="passes over the data; with --steps the run ends at whichever limit comes first")
            p.add_argument("--auto-caption", action="store_true", help="caption images that have no caption first (settings: --param autocaption.model_id=...)")
            p.add_argument("--save-every-epoch", type=int, default=DEFAULTS.save_every_epoch, help="also save after every this many epochs (0 = off)")
            p.add_argument("--lr", type=float, default=DEFAULTS.lr)
            p.add_argument("--schedule", default=DEFAULTS.schedule, choices=("constant", "cosine"), help="learning rate over the run: constant, or a cosine decay to 5%% of --lr over the whole run")
            p.add_argument("--save-probe", action="store_true", help="before each save, run one forward with the live adapter and keep it in <file>.probe.safetensors "
                           "(python -m next.accept save-probe checks the saved file against it)")
            p.add_argument("--warmup", type=int, default=DEFAULTS.warmup, help="optimizer steps over which the learning rate ramps up linearly from 0 (0 = off)")
            p.add_argument("--regions", type=int, default=1, help="timesteps per image in one step, one from each equal part of the range, gradients of the parts "
                           "scaled to equal length before they are averaged (N times the batch per step; see manual/quality.md, 'Timesteps per image')")
            p.add_argument("--regions-from", type=int, default=DEFAULTS.regions_from, help="with --regions: start with this many timesteps per image and fall to --regions over --regions-span batches "
                           "(the first updates set the direction; see manual/quality.md)")
            p.add_argument("--regions-span", type=int, default=DEFAULTS.regions_span, help="batches over which --regions-from falls to --regions")
            p.add_argument("--resume-every", type=int, default=DEFAULTS.resume_every, help="save the training state (adapter, optimizer, step, data position) to <output>/resume every N steps, "
                           "writing it so a power cut cannot damage it; running the same command again continues from the newest state (0 = off)")
            p.add_argument("--batch-first", type=int, default=DEFAULTS.batch_first, help="a smaller batch size for the first --batch-first-steps batches (e.g. 1 with --batch 4: one picture per update at the start, four at the end); 0 = off")
            p.add_argument("--batch-first-steps", type=int, default=DEFAULTS.batch_first_steps, help="batches that use the --batch-first size")
            p.add_argument("--lora-plus", type=float, default=DEFAULTS.lora_plus, help="LoRA+: the up-projection factor (B) learns this many times faster than the down-projection (A); 1 = off, 16 is the usual suggestion")
            p.add_argument("--clean", type=float, default=DEFAULTS.clean, help="LoRA cleaner (0 = off): the share of the adapter to keep (e.g. 0.15). Gates on the adapter's weights train with the loss "
                           "(and --anchor when on); the highest gates stay and, unless --clean-no-freeze, are frozen while the rest trains from zero. Writes <output>/clean/lora_cleaned.safetensors")
            p.add_argument("--clean-at", type=int, default=DEFAULTS.clean_at, help="step the cleaning happens before (0 with --init-adapter)")
            p.add_argument("--clean-steps", type=int, default=DEFAULTS.clean_steps, help="gate training steps")
            p.add_argument("--clean-mode", choices=("element", "component", "magnitude", "product", "singular"), default=DEFAULTS.clean_mode,
                           help="element / component: gates per weight / per singular direction, trained with the loss. Without training: magnitude (the largest weights stay), product (the weights "
                                "contributing most to the change), singular (the largest singular directions of each layer)")
            p.add_argument("--clean-scope", choices=("global", "layer", "part"), default=DEFAULTS.clean_scope, help="what --clean counts over: the whole adapter, each layer, or each kind of layer")
            p.add_argument("--clean-drop", default=DEFAULTS.clean_drop, help="comma-separated parts (attn1, attn2, ff, proj_in, proj_out, ...) or regions (input_blocks, middle_block, output_blocks) removed completely")
            p.add_argument("--clean-rescale", action="store_true", default=DEFAULTS.clean_rescale, help="scale each layer after the cut so its change keeps its size")
            p.add_argument("--stage", dest="stages", action="append", metavar="STEPS[@LR][:KEEP[/MODE]]",
                           help="one stage of a staged run, repeated: STEPS at learning rate LR (default --lr), then cleaned to KEEP of the adapter (MODE magnitude unless given) "
                                "and what is kept stays frozen. KEEP is the frozen share of the whole adapter in total. The run's steps are the sum "
                                "(e.g. --stage 600:0.08 --stage 400@5e-5:0.15 --stage 300@2e-5:0.3)")
            p.add_argument("--stage-warmup", type=int, default=DEFAULTS.stage_warmup, help="steps each stage after the first ramps its rate up from zero")
            p.add_argument("--clean-lr", type=float, default=DEFAULTS.clean_lr, help="learning rate of the gates")
            p.add_argument("--clean-sparsity", type=float, default=DEFAULTS.clean_sparsity, help="pull toward closed gates (0 = only the loss)")
            p.add_argument("--clean-no-freeze", dest="clean_freeze", action="store_false", default=DEFAULTS.clean_freeze, help="train everything after cleaning instead of freezing the kept part")
            p.add_argument("--anchor", type=float, default=DEFAULTS.anchor, help="anchor loss weight (0 = off): the adapter is penalised for changing the base model's prediction under prompts "
                           "unrelated to the dataset (needs the plain optimizer and no graph; about one more forward and backward per anchor step)")
            p.add_argument("--anchor-every", type=int, default=DEFAULTS.anchor_every, help="steps between anchor passes")
            p.add_argument("--anchor-prompts", default=DEFAULTS.anchor_prompts, help="anchor prompts separated by | (default: a built-in list of eight)")
            p.add_argument("--init-adapter", default=DEFAULTS.init_adapter, help="continue an adapter made earlier (an absolute path, or a name under the loras folder): same rank, targets and kind; the "
                           "optimizer, schedule and step count start fresh")
            p.add_argument("--ema", default=DEFAULTS.ema, help="Adapter EMA window widths, comma-separated (e.g. 0.05,0.1): besides each saved file, one more per width with _emaW in the name")
            p.add_argument("--gradient-policy", choices=("mean", "equal", "project", "common"), default=None,
                           help="how the regions of a batch are reconciled: mean (one loss), equal (unit-length gradients, the default with --regions >= 2), project, common")
            p.add_argument("--policy-floor", type=float, default=DEFAULTS.policy_floor, help="with common: skip the update when the regions' shared direction keeps less than this share (0 = never)")
            p.add_argument("--policy-calibrate-every", type=int, default=DEFAULTS.policy_calibrate_every,
                           help="with equal, --time-balance gradnorm and 3 regions: run the regions alone only every Nth step (0 = every step)")
            p.add_argument("--policy-no-normalize", dest="policy_normalize", action="store_false", help="do not scale each region's gradient to unit length first")
            p.add_argument("--trend-boost", type=float, default=DEFAULTS.trend_boost, help="Weight 1 + B * z on a region whose part of the timestep range is doing better than usual (needs --regions >= 2; 0 = off)")
            p.add_argument("--trend-mode", choices=("trend", "current"), default=DEFAULTS.trend_mode, help="trend: the loss of the region's bins has recently been falling; current: the region's own loss is below the slow average")
            p.add_argument("--trend-penalty", type=float, default=DEFAULTS.trend_penalty, help="Also lower the weight of a region doing worse than usual (0 = nothing is taken away)")
            p.add_argument("--aligned-init", type=int, default=DEFAULTS.aligned_init, help="Set the adapter's A factors from the loss gradient of the first N batches before training (LoRA-GA style; those batches are consumed); 0 = off")
            p.add_argument("--tag-swap", action="store_true", help="train each epoch on one random tag of every caption instead of the whole caption (an experiment)")
            p.add_argument("--rank", type=int, default=DEFAULTS.rank)
            p.add_argument("--optimizer", default=DEFAULTS.optimizer, choices=("AdamWNode", "AdafactorNode", "CAMENode"))
            p.add_argument("--fused", action="store_true")
            p.add_argument("--save-every", type=int, default=DEFAULTS.save_every)
            p.add_argument("--no-checkpointing", dest="checkpointing", action="store_false")
            p.add_argument("--persist-conditioning", action="store_true",
                           help="keep text encodings of stable prompts on disk next to the dataset's latents")
            p.add_argument("--max-shapes", type=int, default=DEFAULTS.max_shapes,
                           help="most distinct padded shapes the model will see (0 = no limit); each new shape costs seconds and driver memory")
            p.add_argument("--batch-area", type=float, default=DEFAULTS.batch_area,
                           help="a batch holds at most batch size times this many base-resolution areas in total (e.g. 1.5), so bigger samples run in smaller batches; 0 = no limit")
            p.add_argument("--pad-fill", dest="fill", default=DEFAULTS.fill, choices=("zero", "mirror"),
                           help="what the padded part of the network input holds: a blank strip, or a reflection of the sample's own noisy input")
            p.add_argument("--pack", type=float, default=DEFAULTS.pack,
                           help="fill short batches by moving left-over samples into larger shapes when no sample is more than this share padding; 0 = off")
            p.add_argument("--graph", action="store_true", help="replay forward and backward from captured device graphs (XPU; needs no dropout, no fused optimizer)")
            p.add_argument("--time-balance", default=DEFAULTS.time_balance, choices=("off", "normalize", "speed", "dro", "gradnorm"),
                           help="weight the loss by how each third of the timestep range is improving; gradnorm: by the inverse of the typical gradient norm of the third (batch 1, one timestep per step)")
            p.add_argument("--balance-sampling", type=float, default=DEFAULTS.balance_sampling,
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


def _with_addons(doc, args):
    if not getattr(args, "addons", None):
        return doc
    try:
        return load_set(doc, NodeRegistry.default(), args.addons)
    except (AddonError, OSError, ValueError) as error:
        raise SystemExit(f"--addons {args.addons}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    set_perf_env()
    args = _parser().parse_args(argv)
    if args.command == "ingest":
        return _ingest(args)
    if args.command == "graph":
        return _run(_with_addons(_graph_file(args), args), args)
    settings, params = _split_settings(args.set)
    try:
        doc = _starter(args, settings, params)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    doc = _with_addons(doc, args)
    if args.dump:
        Path(args.dump).write_text(json.dumps(doc.as_dict(), indent=2), encoding="utf8")
        return 0
    return _run(doc, args)


if __name__ == "__main__":
    sys.exit(main())
