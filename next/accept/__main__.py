"""``quant-probe``: how far a quantized copy of a real checkpoint's UNet moves, by region.

    python -m next.accept quant-probe --models MODELS --file sd_xl_base_1.0.safetensors --precision nf4 int8 --out probe
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
from .loading import load_denoiser
from .quantization import QuantizationProbe


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
    args = parser.parse_args(argv)
    return quant_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
