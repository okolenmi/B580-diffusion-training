#!/usr/bin/env python3
"""L4 measurement: what do heterogeneous captions cost today, in parts?

`non-square` has exactly one caption, so it cannot speak to hetero-caption
batches at all. This uses `datasets/multi-caption` -- byte-identical
latents, 8 synthetic distinct prompts round-robin (measurement only;
quality is irrelevant) -- and measures the three quantities L4's design
rests on, with no trainer change:

1. Loader yield: samples actually trained per epoch at batch 4 under the
   current (prompt, neg, size) grouping, bucketed and not, vs the
   single-caption baseline. The B3 warning ("groups smaller than the batch
   are dropped") made concrete.
2. Hetero batches never form today: distinct prompts per emitted batch.
3. Text-encode cost: cold encode_prompts() for 1 vs 4 vs 8 distinct prompts
   on the real towers (XPU), and the warm (cached) pass -- i.e. the
   per-step conditioning price of hetero batches once L4 plumbing exists.

Needs `datasets/multi-caption`: byte-identical copy of `non-square` with
8 synthetic prompts round-robin (datasets/ is gitignored, so regenerate):

    cp -r datasets/non-square datasets/multi-caption
    python -c "
    import sqlite3, random
    random.seed(7)
    vocab = [...40 words...]  # any fixed list; seed 7, 12 words x 8 prompts
    ..."
(see git history of this file's introduction for the exact block).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def loader_yield(dataset: str, multiple: int, batch: int,
                 keep_incomplete: bool) -> tuple[int, int, int, int]:
    import random
    from paths import resolve_safe_dataset_path
    from manager.loader import ManagedDatasetLoader
    random.seed(1234)
    loader = ManagedDatasetLoader(
        dataset_root=resolve_safe_dataset_path(dataset), batch_size=batch,
        shuffle=True, keep_incomplete=keep_incomplete,
        shape_bucket_multiple=multiple)
    n_prompts = len({t["prompt"] for t in loader.trajectories})
    seen, nb = 0, 0
    for b in loader:
        # Batches carry ONE prompt by construction (_merge_samples takes
        # samples[0]); hetero batches cannot form under this grouping.
        assert isinstance(b["prompt"], str), type(b["prompt"])
        seen += b["x_t"].shape[0]
        nb += 1
    return seen, nb, n_prompts, len(loader.trajectories)


def main() -> int:
    print("== 1. loader yield: samples trained per epoch, batch 4, shuffle ==")
    print(f"{'dataset':>14} {'bucket':>6} {'keep_inc':>8} "
          f"{'trained':>8} {'batches':>8} {'captions':>9}")
    for ds in ("non-square", "multi-caption"):
        for m in (0, 32):
            for ki in (False, True):
                seen, nb, np_, total = loader_yield(ds, m, 4, ki)
                print(f"{ds:>14} x{m:<5} {str(ki):>8} "
                      f"{seen:>8}/{total} {nb:>8} {np_:>9}")

    print("== 2+3. text-encode cost, real towers, XPU ==")
    import torch
    from nodes.core import ExecutionContext
    from nodes.model.checkpoint_loader import SafetensorsCheckpointNode
    from nodes.model.text_encoder import SDXLTextEncoderNode
    from nodes.model.text_encoder_cache import CachingTextEncoderNode
    ctx = ExecutionContext()
    weights = SafetensorsCheckpointNode(ctx).build(
        path="div_4.safetensors")["weights"]
    raw = SDXLTextEncoderNode(ctx).build(weights=weights)["encoder"]
    enc = CachingTextEncoderNode(ctx).build(encoder=raw)["encoder"]

    import sqlite3
    con = sqlite3.connect(REPO / "datasets/multi-caption/metadata.db")
    prompts = [r[0] for r in con.execute(
        "select distinct prompt from trajectories order by prompt")]
    con.close()
    assert len(prompts) == 8, len(prompts)

    def sync():
        if hasattr(torch, "xpu"):
            torch.xpu.synchronize()

    # Throwaway warm-up first: the first encode on a fresh process pays
    # one-time device init (~800 ms), which is not a per-prompt cost and
    # must not land in any prompt's number.
    enc.encode_prompts(["warm up the device"])
    for k in (1, 4, 8):
        # Fresh cache view per k: evict by rebuilding the caching layer
        # around the same towers, so every timed call is a true miss.
        enc_k = CachingTextEncoderNode(ctx).build(encoder=raw)["encoder"]
        ps = prompts[:k]
        sync()
        t0 = time.perf_counter()
        outs = enc_k.encode_prompts(ps)
        sync()
        cold = time.perf_counter() - t0
        sync()
        t0 = time.perf_counter()
        outs2 = enc_k.encode_prompts(ps)
        sync()
        warm = time.perf_counter() - t0
        same = all(torch.equal(a[0].cpu(), b[0].cpu())
                   for a, b in zip(outs, outs2))
        print(f"  {k} prompt(s): cold {cold*1000:7.1f} ms "
              f"({cold/k*1000:6.1f} ms/prompt), "
              f"warm {warm*1000:7.1f} ms, cache-consistent={same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
