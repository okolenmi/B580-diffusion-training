"""A9: many distinct prompts vs prewarm -- what actually OOMs, and what the
"pause" is. Real XPU, real SDXL CLIP (fp16, production dtype) from
div_4.safetensors, real BudgetedResourceControlHandle.

The operator's report: with many different prompts, training OOMs when
empty_cache_every_n_steps=100 but survives at 1, and "the prewarmed
steps are emptied as well -- each time I can see a pause".

Two claims to check, in order:

1. THE THEORY (stage 1): device empty_cache() cannot evict warmed
   prompts. CachingTextEncoder._insert_prompt stores
   ctx.detach().cpu() -- host tensors. empty_cache() only returns
   device allocator segments nothing is allocated from. Proven below
   by warming, emptying, and re-requesting: still hits, zero inner
   calls, millisecond timing.

2. THE MECHANISM (stage 2, this file's --matrix): each miss reloads
   CLIP (host->device ~1.5 GB), encodes, and -- when the controller
   releases text_encoder -- offloads it again WITHOUT any reclaim
   (release() moves + synchronizes; only the trainer's interval loop
   calls empty_cache()). Between reclaims, reserved sits elevated and
   every miss stacks its load transient on top, fragmenting as it
   goes. N=1 resets each step; N=100 lets it creep. The "pause" is
   that ~790 ms load+encode+evict cycle per miss, not an emptied
   cache.

Usage:
    # stage 1: theory test + pause/transient numbers (~2-3 min)
    python -u archive/bad-LoRA-investigation/a9_many_prompts_prewarm.py --stage1
    # stage 2: standing x interval matrix in the background
    python -u archive/bad-LoRA-investigation/a9_many_prompts_prewarm.py --matrix
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.components.device import DeviceContext
from nodes.core import ExecutionContext
from nodes.memory.control_handle import BudgetedResourceControlHandle
from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
from nodes.model.text_encoder import SDXLTextEncoderNode
from nodes.model.text_encoder_cache import CachingTextEncoder
from nodes.resource_budget import ResourceBudget

PROMPTS = [f"a photo of object number {i}, dramatic lighting, high detail"
           for i in range(60)]


def stats(ctx):
    s = ctx.memory_stats()
    return s["reserved_mb"], s["allocated_mb"], s.get("peak_reserved_mb", 0.0)


def load_clip(ctx):
    weights = SafetensorsCheckpointNode(ctx).build(
        path="div_4.safetensors")["weights"]
    enc = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    return enc


def stage1():
    ctx = ExecutionContext()
    dev = DeviceContext.for_device("xpu")
    print(f"free {torch.xpu.mem_get_info()[0] / 2**20:.0f} MB", flush=True)

    enc = load_clip(ctx)
    r0, a0, _ = stats(dev)
    print(f"CLIP loaded: reserved={r0:.0f} allocated={a0:.0f}", flush=True)

    rc = BudgetedResourceControlHandle(
        ResourceBudget(vram_budget_mb=1e9, vram_reserve_mb=0.0), device="xpu")
    cached = CachingTextEncoder(enc, max_entries=512)
    cached.bind_resource_control(rc)
    rc.register("text_encoder", cached, offloadable=True)

    # --- warm 3 prompts, unload (the prewarm sequence) ---
    for p in PROMPTS[:3]:
        cached.encode(p, 1, 512, 512)
    cached.unload()
    dev.synchronize()
    r1, a1, _ = stats(dev)
    print(f"after unload: reserved={r1:.0f} allocated={a1:.0f} "
          f"(freed {r0 - r1:.0f} MB reserved)", flush=True)
    n_keys = len(cached._prompt_cache)

    # --- THE THEORY TEST: empty the device cache, re-request ---
    torch.xpu.empty_cache()
    dev.synchronize()
    t = time.monotonic()
    for p in PROMPTS[:3]:
        cached.encode(p, 1, 512, 512)
    dt = (time.monotonic() - t) * 1000 / 3
    print(f"re-request after empty_cache: {dt:.1f} ms/prompt, "
          f"cache keys {n_keys} -> {len(cached._prompt_cache)}", flush=True)
    print("THEORY: " + ("REFUTED -- a miss happened (slow, new keys)"
                         if len(cached._prompt_cache) != n_keys or dt > 200
                         else "CONFIRMED -- still hits; empty_cache() cannot "
                              "touch CPU tensors"), flush=True)

    # --- one miss through the managed-route path: release, then miss ---
    rc.release("text_encoder")
    dev.synchronize()
    r2, _, _ = stats(dev)
    dev.reset_peak_stats()
    t = time.monotonic()
    rc.ensure_loaded("text_encoder")
    dev.synchronize()
    r3, _, _ = stats(dev)
    t_load = (time.monotonic() - t) * 1000
    t = time.monotonic()
    cached.encode(PROMPTS[3], 1, 512, 512)
    dev.synchronize()
    t_enc = (time.monotonic() - t) * 1000
    _, _, peak = stats(dev)
    print(f"miss cycle: reload={t_load:.0f} ms (+{r3 - r2:.0f} MB), "
          f"encode={t_enc:.0f} ms, step peak reserved={peak:.0f} MB",
          flush=True)
    t = time.monotonic()
    rc.release("text_encoder")
    dev.synchronize()
    t_evict = (time.monotonic() - t) * 1000
    r4, _, _ = stats(dev)
    print(f"evict={t_evict:.0f} ms, reserved after release={r4:.0f} MB "
          f"(no reclaim -- still near peak)", flush=True)
    torch.xpu.empty_cache()
    dev.synchronize()
    r5, _, _ = stats(dev)
    print(f"after empty_cache: reserved={r5:.0f} MB "
          f"(reclaimed {r4 - r5:.0f} MB)", flush=True)

    del cached, enc, rc
    gc.collect()
    torch.xpu.empty_cache()
    print("STAGE1 DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--matrix", action="store_true")
    args = ap.parse_args()
    if args.stage1:
        stage1()
    elif args.matrix:
        from a9_matrix import run_matrix  # noqa: E402
        run_matrix()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
