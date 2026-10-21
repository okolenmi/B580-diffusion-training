"""Device time of int8 against native linear layers, at the shapes an SDXL step runs.

    python scripts/bench_int8_layer.py --batch 3 --size 512            # on the XPU, needs xpu-kernels
    python scripts/bench_int8_layer.py --device cpu --size 64 --iters 2   # smoke test of the script itself

For each layer shape: forward and forward+backward (input gradient only, as in LoRA training) with the native bf16 weight
and with the int8 weight, plus the pieces of the int8 path (rotate, the old dense-weight backward) so the slow part is
named. Times are synchronised wall times per call in milliseconds; the layer counts are approximate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from next.families.general.quantize.backends import make_backend, shape_problem  # noqa: E402
from next.families.general.quantize.ops import Int8LinearFn  # noqa: E402

# (name, tokens at 1/8 latent scale divided by this per side, in features, out features, layers like it in the UNet)
LAYERS = (
    ("640 attention projection", 2, 640, 640, 8 * 4 + 8 * 2),
    ("640 feed-forward in", 2, 640, 5120, 4 * 2),
    ("640 feed-forward out", 2, 2560, 640, 4 * 2),
    ("1280 attention projection", 4, 1280, 1280, 40 * 4 + 40 * 2),
    ("1280 feed-forward in", 4, 1280, 10240, 40),
    ("1280 feed-forward out", 4, 5120, 1280, 40),
    ("cross-attention key/value 1280", 4, 2048, 1280, 40 * 2),
    ("cross-attention key/value 640", 2, 2048, 640, 8 * 2),
)


def timed(fn, device: torch.device, iters: int) -> float:
    sync = getattr(getattr(torch, device.type, None), "synchronize", lambda: None)
    for _ in range(max(2, iters // 4)):
        fn()
    sync()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - start) / iters * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--size", type=int, default=512, help="image side in pixels")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--backend", default="auto", choices=("auto", "xpu_kernels", "torch"))
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    backend = make_backend(args.backend, device)
    print(f"backend {backend.name}, batch {args.batch}, {args.size}px\n")
    head = f"{'layer':<34}{'rows':>7}{'fwd nat':>9}{'fwd int8':>10}{'f+b nat':>9}{'f+b int8':>10}{'rotate':>8}{'old dx':>8}{'new dx':>8}{'x count':>8}"
    print(head)
    totals = {"nat": 0.0, "int8": 0.0}
    for name, divide, k, n, count in LAYERS:
        if shape_problem(n, k):
            print(f"{name:<34}stays native: {shape_problem(n, k)}")
            continue
        side = args.size // 8 // divide
        rows = args.batch * side * side
        weight = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
        w8, scale = backend.prepare(weight)
        x = torch.randn(rows, k, device=device, dtype=torch.bfloat16)
        dy = torch.randn(rows, n, device=device, dtype=torch.bfloat16)
        xg = x.clone().requires_grad_(True)

        def forward_native():
            torch.nn.functional.linear(x, weight)

        def forward_int8():
            with torch.no_grad():
                backend.linear(x, w8, scale, None)

        def both_native():
            torch.nn.functional.linear(xg, weight).backward(dy)

        def both_int8():
            Int8LinearFn.apply(xg, w8, scale, None, backend).backward(dy)

        rotate = timed(lambda: backend.rotate(x), device, args.iters)
        old = timed(lambda: dy @ backend.dequant(w8, scale), device, args.iters)
        new = timed(lambda: backend.dx(dy, w8, scale), device, args.iters)
        row = [timed(f, device, args.iters) for f in (forward_native, forward_int8, both_native, both_int8)]
        totals["nat"] += row[2] * count
        totals["int8"] += row[3] * count
        print(f"{name:<34}{rows:>7}" + "".join(f"{v:>9.2f}" if i != 1 and i != 3 else f"{v:>10.2f}" for i, v in enumerate(row))
              + f"{rotate:>8.2f}{old:>8.2f}{new:>8.2f}{count:>8}")
    print(f"\nforward+backward over these layers, counted by how many of each the UNet has: native {totals['nat']:.0f} ms, int8 {totals['int8']:.0f} ms"
          " (a checkpointed block runs its forward twice, so the real gap is a little larger)")


if __name__ == "__main__":
    main()
