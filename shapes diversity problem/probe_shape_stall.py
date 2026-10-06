#!/usr/bin/env python3
"""Why does a multi-resolution dataset train at half speed on the B580?

Run from the repository root:
    python3 scripts/probe_shape_stall.py --dataset datasets/non-square --batch 2 --passes 3 --csv probe.csv
    python3 scripts/probe_shape_stall.py --shapes 64x96,96x64,72x88 --config tiny --device cpu   # logic check

It runs the project's own SDXL UNet (random weights) with LoRA-like trainable
adapters on every attention/FF Linear, so forward AND backward kernels look
like real training, over the dataset's latent shapes in mixed order, for
several passes. Per step it records wall time, whether this shape was seen
before in this process, CPU time / wall time (about 1.0 during a stall means
ONE thread is busy -- single-threaded JIT work; the GPU is waiting), and the
allocator counters. Then it prints a verdict:

  one-time compile   pass 2+ is as fast as steady state     -> persistent caches
                     + pre-warm fix it; no data change needed
  recurring cost     pass 2+ still slow after every shape   -> cache thrash or
                     was seen                                  allocator; fewer
                                                               shapes needed
  allocator          alloc retries / reserved growth in the -> memory pressure,
                     slow steps                                not compile

Experiments that separate the causes (run each, compare 'pass 2+ mean'):
  A  baseline
  B  ONEDNN_PRIMITIVE_CACHE_CAPACITY=65536 python3 ... (bigger primitive cache)
  C  SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=~/.cache/sycl_kernels, run TWICE:
     the second process shows how much of 'pass 1' was compile
  D  --warm-threads 1 then --warm-threads 2: total warm time shows whether
     compiling shapes concurrently helps (watch device memory)
  E  ONEDNN_VERBOSE=1 ... 2> verbose.txt  (count primitive creations per pass)

Nothing here changes training. It needs no dataset images, only the shape
histogram from metadata.db.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sqlite3
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Run as scripts/probe_shape_stall.py: put the repository root (not scripts/) on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ENV_OF_INTEREST = (
    "ONEDNN_PRIMITIVE_CACHE_CAPACITY", "ONEDNN_VERBOSE", "SYCL_CACHE_PERSISTENT", "SYCL_CACHE_DIR",
    "SYCL_CACHE_IN_MEM", "SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD", "UR_L0_USE_RELAXED_ALLOCATION_LIMITS",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "IGC_EnableDPEmulation", "PYTORCH_XPU_ALLOC_CONF",
)


def parse_shapes(text: str) -> list[tuple[int, int]]:
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        if part:
            h, _, w = part.partition("x")
            out.append((int(h), int(w)))
    return out


def read_dataset_shapes(path: str) -> list[tuple[int, int]]:
    """One entry per distinct (latent_h, latent_w) in a v2 dataset (read-only)."""
    db = Path(path) / "metadata.db"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT DISTINCT latent_h, latent_w FROM trajectories "
                           "WHERE latent_h > 0 AND latent_w > 0").fetchall()
    finally:
        con.close()
    return [(int(h), int(w)) for h, w in rows]


def build_model(config: str, device: str, dtype, use_checkpoint: bool):
    import torch
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W

    cfg = dict(W.SDXL_CONFIG)
    cfg["use_checkpoint"] = use_checkpoint
    if config == "tiny":   # same classes and wiring, small enough for a CPU logic check
        cfg.update(model_channels=32, num_head_channels=16, transformer_depth=[0, 0, 1, 1, 1, 1],
                   transformer_depth_middle=1, transformer_depth_output=[0, 0, 0, 1, 1, 1, 1, 1, 1],
                   context_dim=64, adm_in_channels=96)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)          # create directly in the target dtype, never in fp32
    try:
        with torch.device(device):
            model = UNetModel(**cfg)
    finally:
        torch.set_default_dtype(prev)
    model.requires_grad_(False)
    return model, cfg


def add_lora_like(model, rank: int = 16):
    """Trainable low-rank paths on every Linear inside a transformer block, via
    forward hooks: gives real backward-weight matmuls without the project's
    injector. Returns the trainable parameters."""
    import torch
    params = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ("attn" in name or "ff" in name):
            down = torch.nn.Parameter(torch.randn(rank, mod.in_features, device=mod.weight.device, dtype=mod.weight.dtype) * 0.01)
            up = torch.nn.Parameter(torch.zeros(mod.out_features, rank, device=mod.weight.device, dtype=mod.weight.dtype))
            params += [down, up]
            mod.register_forward_hook(lambda m, i, o, d=down, u=up: o + (i[0] @ d.t()) @ u.t())
    return params


class Device:
    def __init__(self, name: str):
        import torch
        self.torch, self.name = torch, name
        self.mod = getattr(torch, name, None) if name != "cpu" else None

    def sync(self):
        if self.mod is not None and hasattr(self.mod, "synchronize"):
            self.mod.synchronize()

    def counters(self) -> dict:
        out = dict(reserved_mb=None, retries=None)
        if self.mod is None:
            return out
        try:
            out["reserved_mb"] = self.mod.memory_reserved() / 2**20
        except Exception:
            pass
        try:
            stats = self.mod.memory_stats()
            out["retries"] = stats.get("num_alloc_retries")
        except Exception:
            pass
        return out


def step(model, params, shape, batch, cfg, device, dtype, rng):
    """One training-like step: forward, loss, gradient w.r.t. the adapters only
    (autograd.grad, so concurrent warm-up threads never touch .grad)."""
    import torch
    h, w = shape
    kw = dict(device=device, dtype=dtype)
    x = torch.randn(batch, 4, h, w, **kw)
    t = torch.randint(0, 1000, (batch,), device=device)
    ctx = torch.randn(batch, 77, cfg["context_dim"], **kw)
    y = torch.randn(batch, cfg["adm_in_channels"], **kw)
    out = model(x, t, ctx, y)
    loss = out.float().pow(2).mean()
    torch.autograd.grad(loss, params)


def run(args) -> int:
    import torch
    from nodes.xpu_env import set_xpu_perf_env_vars
    if not args.no_env:
        set_xpu_perf_env_vars()
    device = args.device
    if device == "auto":
        device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    if device == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    shapes = parse_shapes(args.shapes) if args.shapes else read_dataset_shapes(args.dataset)
    if not shapes:
        print("no shapes", file=sys.stderr)
        return 2
    dev = Device(device)
    print(f"device={device} dtype={dtype} config={args.config} batch={args.batch} "
          f"distinct shapes={len(shapes)} passes={args.passes} checkpointing={not args.no_checkpoint}")
    print("environment:", {k: os.environ[k] for k in ENV_OF_INTEREST if k in os.environ} or "(none of the interesting variables set)")
    model, cfg = build_model(args.config, device, dtype, not args.no_checkpoint)
    params = add_lora_like(model)
    rng = random.Random(args.seed)
    seen: set[tuple[int, int]] = set()

    if args.warm_threads:
        order = sorted(shapes, key=lambda s: -s[0] * s[1])      # largest first: sets the allocator's high-water mark early
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.warm_threads) as pool:
            list(pool.map(lambda s: (step(model, params, s, args.batch, cfg, device, dtype, rng), dev.sync()), order))
        seen.update(shapes)
        print(f"warm-up of {len(order)} shapes with {args.warm_threads} thread(s): {time.perf_counter() - t0:.1f} s "
              f"(device reserved {dev.counters()['reserved_mb']} MB)")

    rows = []
    for p in range(1, args.passes + 1):
        order = list(shapes)
        rng.shuffle(order)
        prev = None
        for i, shape in enumerate(order):
            for rep in range(args.repeat):       # --repeat > 1 mimics clumps of identical shapes
                c0 = dev.counters(); dev.sync()
                w0, p0 = time.perf_counter(), time.process_time()
                step(model, params, shape, args.batch, cfg, device, dtype, rng)
                dev.sync()
                wall, cpu = time.perf_counter() - w0, time.process_time() - p0
                c1 = dev.counters()
                rows.append(dict(
                    pass_no=p, idx=i, rep=rep, shape=f"{shape[0]}x{shape[1]}", first_seen=int(shape not in seen),
                    same_as_prev=int(prev == shape), wall_s=round(wall, 4), cpu_frac=round(cpu / wall, 2) if wall else 0,
                    reserved_mb=c1["reserved_mb"], retries_delta=(None if c0["retries"] is None or c1["retries"] is None
                                                                  else c1["retries"] - c0["retries"])))
                seen.add(shape)
                prev = shape
        done = [r for r in rows if r["pass_no"] == p]
        print(f"pass {p}: {len(done)} steps, total {sum(r['wall_s'] for r in done):.1f} s, "
              f"mean {statistics.mean(r['wall_s'] for r in done):.3f} s/step")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)
        print(f"wrote {args.csv}")
    verdict(rows, args)
    return 0


def verdict(rows, args) -> None:
    """Compare three kinds of step. A *repeat* (same shape as the step just
    before it) is always warm, so it is the steady-state reference. A *first
    sighting* pays any one-time cost. A *revisit* (a shape seen earlier, but
    not the previous step) is the one that tells the causes apart: fast means
    the cost was one-time, slow means something is evicted or re-specialised
    every time."""
    def med(sel):
        v = [r["wall_s"] for r in sel]
        return statistics.median(v) if v else float("nan")
    repeats = [r for r in rows if r["same_as_prev"] and not r["first_seen"]]
    firsts = [r for r in rows if r["first_seen"] and not r["same_as_prev"]]
    revisits = [r for r in rows if not r["first_seen"] and not r["same_as_prev"]]
    print("\n--- verdict " + "-" * 56)
    if not repeats or not revisits:
        print("need --repeat >= 2 and --passes >= 2 to separate one-time from recurring cost")
        return
    steady = med(repeats)
    first_x = med(firsts) / steady if firsts else float("nan")
    rev_x = med(revisits) / steady
    slow = [r for r in revisits if r["wall_s"] > 1.5 * steady]
    slow_frac = len(slow) / len(revisits)
    print(f"warm repeat (same shape as previous step) n={len(repeats):3d}: median {steady:.3f} s   <- steady state")
    print(f"first sighting of a shape                 n={len(firsts):3d}: median {med(firsts):.3f} s = {first_x:.1f}x steady")
    print(f"revisit of an already-seen shape          n={len(revisits):3d}: median {med(revisits):.3f} s = {rev_x:.2f}x steady; "
          f"{slow_frac:.0%} of revisits are >1.5x steady")
    cpu_slow = [r["cpu_frac"] for r in rows if r["wall_s"] > 1.5 * steady]
    if cpu_slow:
        print(f"CPU time / wall time during slow steps: median {statistics.median(cpu_slow):.2f} "
              f"(about 1.0 = one thread busy while the device waits; the project saw ~30% GPU)")
    have_retries = rows[0]["retries_delta"] is not None
    retries = sum(r["retries_delta"] or 0 for r in rows)
    print(f"allocator retries over the whole run:   {retries if have_retries else 'not reported by this torch build'}")
    print()
    if slow_frac < 0.10 and rev_x < 1.25:
        if first_x > 1.5:
            print("VERDICT: ONE-TIME cost. A shape is slow the first time it appears and normal afterwards.")
            print("  -> persistent kernel caches (SYCL_CACHE_PERSISTENT) and a pre-warm pass remove it;")
            print("     no change to the dataset is needed.")
        else:
            print("VERDICT: no shape-dependent cost measured. Look elsewhere (data pipeline, memory controller).")
    elif have_retries and retries > 0:
        print("VERDICT: RECURRING cost with allocator retries -> memory pressure / fragmentation, not compile.")
        print("  -> fewer distinct shapes helps; so does headroom (budget, checkpointing).")
    else:
        print("VERDICT: RECURRING cost without allocator retries -> revisited shapes are re-specialised, i.e. a cache is")
        print("  too small or being evicted. Re-run with ONEDNN_PRIMITIVE_CACHE_CAPACITY=65536: if revisits become")
        print("  fast, raise the capacity by default. If not, fewer distinct shapes is the only lever.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset"); src.add_argument("--shapes", help='"HxW,HxW" latent px')
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=2, help="steps per shape per visit (clump size)")
    ap.add_argument("--config", choices=("sdxl", "tiny"), default="sdxl")
    ap.add_argument("--device", choices=("auto", "xpu", "cpu"), default="auto")
    ap.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--warm-threads", type=int, default=0)
    ap.add_argument("--no-env", action="store_true", help="do not call set_xpu_perf_env_vars()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
