"""Command line of the lab.

    python -m next.lab template > experiment.json
    python -m next.lab run experiment.json [--phase train render report] [--arms name ...] [--force]
    python -m next.lab render --models M --file CKPT --lora a.safetensors --prompt "..." --seeds 1 2 --strengths 0.5 1 --out DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .experiment import Experiment
from . import setup as lab_setup
from .templates import TEMPLATES
from .render import Grid, ImageStore, Renderer
from .report import Report
from .runner import REPO_ROOT, Lab

PHASES = ("train", "render", "report")


def backend_factory(experiment: Experiment):
    from .pipeline import SDXLBackend

    return lambda lora: SDXLBackend(experiment.models, experiment.checkpoint, experiment.device, lora)


def run_one(path: str, args) -> None:
    experiment = Experiment.load(path)
    lab = Lab(experiment, backend_factory(experiment))
    arms = [experiment.arm(n) for n in args.arms] if args.arms else list(experiment.arms)
    phases = args.phase or list(PHASES)
    if "train" in phases:
        lab.train(arms, args.force)
    if "render" in phases:
        lab.render(arms, args.force)
    if "report" in phases:
        report = Report(experiment)
        path = report.write()
        print(f"report: {path}\nshare:  {report.pack()}")


def refuse_unusable(paths: list[str]) -> None:
    """Every file is checked before the first one starts, so a placeholder path or a missing adapter costs nothing."""
    problems = []
    for path in paths:
        try:
            found = lab_setup.check_experiment(Experiment.load(path))
        except (OSError, ValueError, KeyError) as error:
            found = [f"cannot be read: {error}"]
        problems += [f"{path}: {line}" for line in found]
    if problems:
        raise SystemExit("not started:\n  " + "\n  ".join(problems))


def run_in_process(path: str, args) -> int:
    """One file in a process of its own: whatever the render phase left on the card (allocator cache, driver state) goes away with the process,
    so the next file's training starts on an empty card."""
    command = [sys.executable, "-m", "next.lab", "run", path, *(["--phase", *args.phase] if args.phase else []), *(["--force"] if args.force else [])]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])])}
    return subprocess.run(command, env=env).returncode


def run(args) -> int:
    """One experiment file, or a queue of them: each runs to its report in its own process, a failed one is reported and the queue goes on.
    Running the same command again after an interruption skips what is done and continues the arm that was cut."""
    if args.arms and len(args.experiment) > 1:
        raise SystemExit("--arms names arms of one experiment; give one file with it")
    refuse_unusable(args.experiment)
    if len(args.experiment) == 1:
        run_one(args.experiment[0], args)
        return 0
    failed = []
    for path in args.experiment:
        print(f"=== {path}", flush=True)
        if run_in_process(path, args):
            print(f"{path}: failed", flush=True)
            failed.append(path)
    if failed:
        print(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


def render(args) -> int:
    from .pipeline import SDXLBackend

    grid = Grid(prompts=tuple(args.prompt or ("",)), seeds=tuple(args.seeds), strengths=tuple(args.strengths), steps=args.steps, cfg=args.cfg,
                sampler=args.sampler, scheduler=args.scheduler, width=args.width, height=args.height)
    for lora in args.lora or [None]:
        store = ImageStore(args.out)
        backend = SDXLBackend(args.models, args.file, args.device, lora)
        Renderer(backend, device=args.device).render(grid.recipes() if lora else grid.recipes((0.0,)), {"lab": 1, "adapter": lora, "checkpoint": args.file}, on_image=store.save)
    return 0


def engines(args) -> int:
    from ..accept.loading import checkpoint_path
    from .engines import EngineComparison
    from .pipeline import SDXLBackend

    grid = Grid(prompts=tuple(args.prompt or ("", "a portrait of a woman, detailed face")), seeds=tuple(args.seeds), strengths=(1.0,) if args.lora else (0.0,), steps=args.steps,
                cfg=args.cfg, sampler=args.sampler, scheduler=args.scheduler, width=args.width, height=args.height)
    out = Path(args.out)
    store = ImageStore(out / "ours")
    Renderer(SDXLBackend(args.models, args.file, args.device, args.lora), device=args.device).render(grid.recipes(), {"lab": 1, "adapter": args.lora}, on_image=store.save)
    comparison = EngineComparison(grid.recipes(), out / "ours", out / "comfy", args.comfy, args.comfy_python or sys.executable, checkpoint_path(args.models, args.file),
                                  args.lora, args.comfy_dtype, args.comfy_device)
    comparison.render_theirs()
    text = comparison.markdown()
    (out / "engines.md").write_text(text, encoding="utf8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m next.lab")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("template", help="print an experiment file to start from")
    t.add_argument("kind", nargs="?", default="ranges", choices=tuple(TEMPLATES))
    t.add_argument("--setup", help=f"your setup file (default ./{lab_setup.DEFAULT_FILE}): models, datasets, checkpoint, images, output, device and saved adapters")
    sub.add_parser("setup", help="print a setup file to fill in once; every template reads it")
    p = sub.add_parser("run", help="train the arms, render and measure them, write the report")
    p.add_argument("experiment", nargs="+", help="one experiment file, or several to run one after the other (the command can be run again after an interruption: it continues)")
    p.add_argument("--phase", nargs="*", choices=PHASES)
    p.add_argument("--arms", nargs="*")
    p.add_argument("--force", action="store_true", help="redo what is already there")
    r = sub.add_parser("render", help="render a grid with one or more adapter files (no adapter when none is given)")
    r.add_argument("--models", required=True)
    r.add_argument("--file", required=True)
    r.add_argument("--lora", nargs="*")
    r.add_argument("--prompt", action="append")
    r.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    r.add_argument("--strengths", type=float, nargs="+", default=[0.5, 1.0])
    r.add_argument("--steps", type=int, default=25)
    r.add_argument("--cfg", type=float, default=6.0)
    r.add_argument("--sampler", default="dpmpp_2m", choices=("euler", "dpmpp_2m"))
    r.add_argument("--scheduler", default="normal", choices=("normal", "simple", "karras"))
    r.add_argument("--width", type=int, default=1024)
    r.add_argument("--height", type=int, default=1024)
    r.add_argument("--device", default="xpu")
    r.add_argument("--out", default="lab-render")
    e = sub.add_parser("engines", help="render the same recipes with this project and with ComfyUI and compare the images")
    for flag, kwargs in (("--models", {"required": True}), ("--file", {"required": True}), ("--comfy", {"required": True}), ("--comfy-python", {}), ("--lora", {}),
                         ("--comfy-dtype", {"default": "auto", "choices": ("auto", "bf16", "fp16", "fp32")}), ("--comfy-device", {"default": "auto", "choices": ("auto", "cpu")}),
                         ("--seeds", {"type": int, "nargs": "+", "default": [1, 2]}), ("--steps", {"type": int, "default": 20}), ("--cfg", {"type": float, "default": 6.0}),
                         ("--sampler", {"default": "dpmpp_2m", "choices": ("euler", "dpmpp_2m")}), ("--scheduler", {"default": "normal", "choices": ("normal", "simple", "karras")}),
                         ("--width", {"type": int, "default": 1024}), ("--height", {"type": int, "default": 1024}), ("--device", {"default": "xpu"}), ("--out", {"default": "lab-engines"})):
        e.add_argument(flag, **kwargs)
    e.add_argument("--prompt", action="append")
    args = parser.parse_args(argv)
    if args.command == "setup":
        print(json.dumps(lab_setup.skeleton(), indent=1))
        return 0
    if args.command == "template":
        try:
            template = lab_setup.apply(TEMPLATES[args.kind], lab_setup.load(args.setup))
        except lab_setup.SetupError as error:
            raise SystemExit(str(error)) from error
        print(json.dumps(template, indent=1))
        return 0
    return {"run": run, "render": render, "engines": engines}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
