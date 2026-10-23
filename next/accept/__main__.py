"""Instruments and automatic verdicts for the hardware runs.

    python -m next.accept quant-probe --models MODELS --file sd_xl_base_1.0.safetensors --precision nf4 int8 --out probe
    python -m next.accept run speed nf4 --out accept -- --models MODELS --datasets D --file F --dataset NAME --output OUT --batch 2 --resolution 512
    python -m next.accept judge perf-bundle/matrix speed        # a bundle that came back from ``python -m next.perf hwcheck``
    python -m next.accept list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..engine.devices import resolve_device
from ..families.general.quantize.backends import make_backend
from ..families.general.quantize.int8 import Int8Quantizer
from ..families.general.quantize.nf4 import Nf4Quantizer
from ..families.general.quantize.nf4_backends import make_nf4_backend
from .fixtures import DenoiserInputs
from .acceptance import Acceptance, choose
from .loading import load_denoiser
from .quantization import QuantizationProbe
from .suites import SuiteBook
from .verdict import Judge, write_verdict


def quantizer_for(args, device):
    if args.precision_now == "int8":
        return Int8Quantizer(make_backend(args.backend, device), args.min_params, args.int8_grad)
    return Nf4Quantizer(make_nf4_backend(args.backend, device), args.min_params, args.double_quant)


def quant_probe(args) -> int:
    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for precision in args.precision:
        args.precision_now = precision
        print(f"{precision}: loading {args.file} ...", flush=True)
        denoiser = load_denoiser(args.models, args.file, str(device))
        inputs = DenoiserInputs(resolution=args.resolution, timesteps=args.timesteps, device=device, dtype=torch.bfloat16)
        quantizer = quantizer_for(args, device)
        result = QuantizationProbe(limit_mb=args.limit_mb).run(denoiser.module(), quantizer, lambda _: inputs.run(denoiser), precision=precision)
        (out / f"quant-{precision}.md").write_text(result.markdown(), encoding="utf8")
        (out / f"quant-{precision}.json").write_text(json.dumps(result.to_json(), indent=1), encoding="utf8")
        print(result.markdown(), flush=True)
        del denoiser
        getattr(torch, str(device).split(":")[0]).empty_cache()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m next.accept")
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("quant-probe", help="dense against quantized copy of a checkpoint's UNet: weight, output and gradient error by region")
    probe.add_argument("--models", required=True)
    probe.add_argument("--file", required=True, help="checkpoint relative to MODELS/checkpoints (or MODELS)")
    probe.add_argument("--precision", nargs="+", default=["nf4", "int8"], choices=("nf4", "int8"))
    probe.add_argument("--device", default="xpu")
    probe.add_argument("--backend", default="auto", choices=("auto", "xpu_kernels", "torch"))
    probe.add_argument("--resolution", type=int, default=512)
    probe.add_argument("--timesteps", type=int, nargs="+", default=[100, 500, 900])
    probe.add_argument("--min-params", type=int, default=1 << 20)
    probe.add_argument("--int8-grad", default="dense", choices=("dense", "int8"))
    probe.add_argument("--double-quant", action="store_true")
    probe.add_argument("--limit-mb", type=float, default=2048.0, help="host memory for the recorded block outputs and gradients")
    probe.add_argument("--out", default="accept-quant-probe")
    pad = sub.add_parser("padding-sheet", help="a picture of what zero and mirror padding put around samples of a folder of images")
    pad.add_argument("--images", required=True)
    pad.add_argument("--canvas", default="64x64", help="canvas HxW in the units shown (pixels here, latent size times 8 for a real bucket)")
    pad.add_argument("--count", type=int, default=4)
    pad.add_argument("--sigma", type=float, default=0.6)
    pad.add_argument("--seed", type=int, default=0)
    pad.add_argument("--out", default="padding-sheet.png")
    run = sub.add_parser("run", help="run suites (each scenario in its own process) and write verdict.md; train arguments follow --")
    run.add_argument("suites", nargs="*", help="default: all (see 'list')")
    run.add_argument("--out", default="accept-run")
    run.add_argument("--steps", type=int, default=28)
    run.add_argument("--warmup", type=int, default=20)
    run.add_argument("train_args", nargs=argparse.REMAINDER)
    judge = sub.add_parser("judge", help="judge folders that already exist (a hwcheck bundle's matrix/ folder, or an earlier run) without running anything")
    judge.add_argument("root", help="the folder holding one subfolder per scenario")
    judge.add_argument("suites", nargs="*")
    judge.add_argument("--out", default="accept-verdict")
    judge.add_argument("--warmup", type=int, default=20)
    sub.add_parser("list", help="the suites and their checks")
    args = parser.parse_args(argv)
    if args.command == "list":
        for suite in SuiteBook().all().values():
            print(f"{suite.name}  (TESTING {suite.item}): {suite.title}\n  scenarios: {', '.join(suite.scenario_names())}")
            print("".join(f"  - {c.name}\n" for c in suite.checks))
        return 0
    if args.command == "padding-sheet":
        from PIL import Image

        from .visual import PaddingSheet
        files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))[: args.count]
        if not files:
            raise SystemExit(f"no images in {args.images}")
        height, width = (int(v) for v in args.canvas.lower().split("x"))
        print(PaddingSheet((height, width), args.sigma, args.seed).sheet([Image.open(f) for f in files]).save(args.out))
        return 0
    if args.command == "judge":
        book = SuiteBook(args.warmup)
        verdicts = [Judge(args.root).judge(s) for s in choose(args.suites, book)]
        print(write_verdict(verdicts, args.out))
        return 0
    if args.command == "run":
        train = [a for a in args.train_args if a != "--"]
        if not train:
            raise SystemExit("give the training arguments after --, e.g.: run speed -- --models M --datasets D --file F --dataset NAME --output OUT")
        book = SuiteBook(args.warmup, str(Path(args.out) / "ceilings.json"))
        print(Acceptance(choose(args.suites, book), Path(args.out), train, steps=args.steps).run())
        return 0
    return quant_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
