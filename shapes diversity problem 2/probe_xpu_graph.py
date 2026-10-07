#!/usr/bin/env python3
"""L5.3: is XPU graph capture worth it here? A feasibility probe, not an
integration.

The project's own measurement says the training step is bound by one thread
issuing ~22,000 kernel launches per step at ~48 us each
(scripts/count_launches.py; `shapes diversity problem/MEASURED-shape-stall.md`).
That is exactly the workload CUDA/XPU graph capture exists for: replaying a
captured launch sequence instead of issuing it again. If replay removes most of
the launch cost it is worth more than every launch-count micro-optimisation
available, because it is the only one that attacks the whole 22,000 at once.

So: probe it, in isolation, on one real UNet forward+backward at one shape,
with static input buffers, and report the replay speedup. Nothing is built
until this number exists.

**Hazards this probes, and what each would mean if it bites:**

  * **The checkpoint function's RNG.** With `use_checkpoint=True` the backward
    recomputes the forward, and a checkpoint wrapper that saves/restores RNG
    state (`fork_rng` and friends) issues ops that cannot be captured -- or
    capture fine and replay with a frozen RNG. The second is worse than a
    failure, because it is silent, so capture is attempted with the production
    setting rather than with checkpointing off.
  * **Autograd across a replay.** A captured graph has a fixed structure; the
    backward it replays is the one that was captured. Gradients therefore land
    in fixed buffers, which is why this re-zeroes and re-populates the input
    buffers each replay instead of building a fresh graph.
  * **The memory pool.** Capture allocates from a private pool; the reserved
    high-water mark reported here is *after* capture and is not comparable to
    an eager run's, because the pool is retained for replay.

Reported either way. A failure with its exception is a result; it says the
approach does not apply to this configuration, which is worth knowing as much
as a speedup is.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MB = 1024 ** 2


def build(device: str, dtype, rank: int, use_checkpoint: bool):
    """The project's own SDXL UNet with LoRA, random weights. Values are
    irrelevant to launch counts and timings; what matters is that this is the
    same architecture and op sequence the trainer runs."""
    import torch
    from nodes.model.lora import LoRAConfig, inject_lora_into_unet
    from nodes.model.unet import UNetModel
    from nodes.model.unet_wrapper import ComfyUNetWrapper as W
    if use_checkpoint:
        from nodes.model.gradient_checkpointing import FrozenParamSafeCheckpointing
        FrozenParamSafeCheckpointing().apply()

    cfg = dict(W.SDXL_CONFIG)
    cfg["use_checkpoint"] = use_checkpoint
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = UNetModel(**cfg)
    finally:
        torch.set_default_dtype(prev)
    model.requires_grad_(False)
    inject_lora_into_unet(model, LoRAConfig(rank=rank, alpha=1.0))
    params = [p for p in model.parameters() if p.requires_grad]
    return model, params, cfg


def timed(fn, device, repeats: int, warmup: int) -> list[float]:
    """Wall seconds per call, in seconds, after `warmup` untimed calls."""
    import torch
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="capture without activation checkpointing. The "
                         "production setting is checkpointing ON, so this is "
                         "the diagnostic arm: if it captures here and not "
                         "there, the checkpoint wrapper is the blocker")
    ap.add_argument("--sdpa-backend", default=None,
                    choices=["MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION",
                             "CUDNN_ATTENTION"],
                    help="force one scaled_dot_product_attention backend for "
                         "the capture. The first production attempt failed "
                         "inside SDPA with 'Graph nodes cannot depend on "
                         "events from outside the graph', which is an "
                         "event/synchronisation dependency created outside "
                         "the capture -- so which backend is chosen is the "
                         "first thing to find out, not the last")
    ap.add_argument("--math-sdpa-only", action="store_true",
                    help="shorthand for --sdpa-backend MATH")
    ap.add_argument("--device", default="xpu")
    args = ap.parse_args(argv)

    # Order matters here and it was wrong the first time: the XPUGraph
    # availability check below imports torch, and importing torch is what
    # freezes the performance environment. nodes/xpu_env.py's own docstring is
    # explicit that these have to be set *before* torch is imported, and a
    # probe that quietly ran with different settings than the trainer would be
    # comparing two configurations and calling it one.
    from nodes.xpu_env import set_xpu_perf_env_vars
    set_xpu_perf_env_vars()

    import torch
    if not hasattr(torch.xpu, "XPUGraph"):
        print("torch.xpu.XPUGraph is absent on this build -- capture is not "
              "available and the question is closed.")
        return 1

    dtype = torch.bfloat16
    use_ckpt = not args.no_checkpoint
    backend = "MATH" if args.math_sdpa_only else args.sdpa_backend
    sdpa_ctx = None
    if backend:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        sdpa_ctx = sdpa_kernel(getattr(SDPBackend, backend))
        print(f"forcing SDPA backend {backend}")
    print(f"torch {torch.__version__}, XPUGraph present, "
          f"batch {args.batch}, latent {args.latent}x{args.latent}, "
          f"rank {args.rank}, activation checkpointing "
          f"{'ON (production)' if use_ckpt else 'OFF (diagnostic)'}")
    free0, total = torch.xpu.mem_get_info()
    print(f"VRAM before: {free0 / MB:.0f} MB free of {total / MB:.0f} MB")

    model, params, cfg = build(args.device, dtype, args.rank, use_ckpt)
    kw = dict(device=args.device, dtype=dtype)

    # Static input buffers. Allocated once and reused by both arms, so the
    # eager/replay comparison differs only in how the work is issued.
    x = torch.randn(args.batch, 4, args.latent, args.latent, **kw)
    t = torch.randint(0, 1000, (args.batch,), device=args.device)
    ctx = torch.randn(args.batch, 77, cfg["context_dim"], **kw)
    y = torch.randn(args.batch, cfg["adm_in_channels"], **kw)

    def eager_step():
        for p in params:
            p.grad = None
        out = model(x, t, ctx, y)
        out.float().pow(2).mean().backward()

    if sdpa_ctx is not None:
        sdpa_ctx.__enter__()
    try:
        return _measure(args, torch, model, params, cfg, eager_step,
                        (x, t, ctx, y), use_ckpt)
    finally:
        # The forced backend has to cover the eager baseline, the capture AND
        # the replays -- a replay that dispatched to a different backend than
        # the one captured is not a replay of that graph at all. So this is a
        # try/finally around all of it, not an enter before the capture.
        if sdpa_ctx is not None:
            sdpa_ctx.__exit__(None, None, None)


def _measure(args, torch, model, params, cfg, eager_step, inputs,
             use_ckpt) -> int:
    """The measurement proper, split out so the forced-SDPA-backend context
    can wrap all of it in one try/finally (see the call site)."""
    x, t, ctx, y = inputs
    eager = timed(eager_step, args.device, args.repeats, args.warmup)
    eager_median = statistics.median(eager)
    free1, _ = torch.xpu.mem_get_info()
    print(f"\neager: median {eager_median * 1000:.1f} ms/step "
          f"(min {min(eager) * 1000:.1f}, max {max(eager) * 1000:.1f}), "
          f"reserved {torch.xpu.memory_reserved() / MB:.0f} MB, "
          f"free {free1 / MB:.0f} MB")

    # ---- capture ------------------------------------------------------------
    print("\ncapturing one forward+backward into an XPUGraph ...")
    graph = torch.xpu.XPUGraph()
    pool = None
    try:
        pool = torch.xpu.graph_pool_handle()
        # Warm up on a side stream first, as the capture docs require: the
        # first execution of a kernel does lazy setup that cannot be captured.
        side = torch.xpu.Stream()
        side.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(side):
            for _ in range(3):
                eager_step()
        torch.xpu.current_stream().wait_stream(side)
        torch.xpu.synchronize()

        for p in params:
            p.grad = None
        with torch.xpu.graph(graph, pool=pool):
            static_out = model(x, t, ctx, y)
            static_loss = static_out.float().pow(2).mean()
            static_loss.backward()
        torch.xpu.synchronize()
    except Exception as exc:  # noqa: BLE001 -- the failure IS the result
        print(f"CAPTURE FAILED: {type(exc).__name__}: {exc}")
        print()
        print("last frames:")
        for line in traceback.format_exc().splitlines()[-12:]:
            print("   ", line)
        print()
        print(f"eager baseline above still stands: {eager_median * 1000:.1f} "
              f"ms/step, {len(eager)} repeats.")
        free2, _ = torch.xpu.mem_get_info()
        print(f"VRAM after the failed attempt: {free2 / MB:.0f} MB free "
              f"(the capture pool is retained even on failure)")
        return 2

    print("capture succeeded")

    def replay_step():
        graph.replay()

    try:
        replay = timed(replay_step, args.device, args.repeats, args.warmup)
    except Exception as exc:  # noqa: BLE001
        print(f"REPLAY FAILED: {type(exc).__name__}: {exc}")
        return 2
    replay_median = statistics.median(replay)

    free3, _ = torch.xpu.mem_get_info()
    reserved = torch.xpu.memory_reserved() / MB
    print(f"replay: median {replay_median * 1000:.1f} ms/step "
          f"(min {min(replay) * 1000:.1f}, max {max(replay) * 1000:.1f})")
    print(f"reserved {reserved:.0f} MB, free {free3 / MB:.0f} MB")
    print()
    print(f"speedup {eager_median / replay_median:.2f}x "
          f"({eager_median * 1000:.1f} -> {replay_median * 1000:.1f} ms); "
          f"{(1 - replay_median / eager_median) * 100:.0f}% of the step removed")
    print(f"VRAM cost of capture: {(free1 - free3) / MB:.0f} MB "
          f"(the pool is retained for replay, so this is not recoverable)")

    # ---- does it still compute the right thing? ---------------------------
    # A replay that silently reuses a stale buffer or a frozen RNG would still
    # be fast. So: change the input, replay, and check the loss moved.
    before = float(static_loss.detach())
    with torch.no_grad():
        x.copy_(torch.randn_like(x) * 3.0)
    replay_step()
    after = float(static_loss.detach())
    moved = abs(after - before) > 1e-6
    print()
    print(f"input changed then replayed: static loss {before:.6f} -> "
          f"{after:.6f} -- {'MOVED' if moved else 'DID NOT MOVE'}")
    if not moved:
        print("  a replay that ignores its input buffers would look exactly "
              "like a fast one here, so this is checked rather than assumed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
