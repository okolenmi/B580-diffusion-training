"""Re-classify the 300-step run, where there are enough revisits to trust.

The earlier classification used 39 steps and found revisits at 0.95x steady --
but that sample held only 3 revisits, so it could not have detected a 1.7x
penalty. The 300-step run has hundreds. This joins the seeded shape order
against the recorded step times the same way, but over the whole run, and
reports the per-step penalty by shape area rather than as one aggregate.

The area breakdown is the part that decides whether the residual is a
one-time cost wearing a recurrence's clothes: a per-shape constant paid on
every visit predicts a flat penalty across sizes, while a compute cost
predicts a penalty that scales with area.
"""

import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
sys.path.insert(0, str(REPO))


def shape_sequence(dataset, batch=2, seed=1234, need=400):
    random.seed(seed)
    from nodes.dataset.managed import ManagedDatasetSourceNode
    batches = ManagedDatasetSourceNode(None).build(
        dataset_root=dataset, batch_size=batch, shuffle=True,
        keep_incomplete_batches=False)["batches"]
    out = [f"{b['x_t'].shape[-2]}x{b['x_t'].shape[-1]}" for b in batches]
    while len(out) < need:
        out += out
    return out


def load(label):
    rows = [json.loads(l) for l in
            (REPO / "runs" / "hw_validation" / label / "steps.jsonl")
            .read_text().splitlines() if l.strip()]
    t = [r for r in rows if r.get("dt_sec") and not r.get("covers_load")]
    t.sort(key=lambda r: r["step"])
    return t


def area(shape):
    h, w = shape.split("x")
    return int(h) * int(w)


def classify(timed, shapes):
    first_timed = timed[0]["step"]
    seen, prev, out = set(), None, []
    for r in timed:
        idx = r["step"] - first_timed
        s = shapes[idx] if 0 <= idx < len(shapes) else None
        t = r["dt_sec"]
        if s is None:
            out.append((None, t, None))
            continue
        kind = "repeat" if s == prev else ("first" if s not in seen else "revisit")
        out.append((s, t, kind))
        seen.add(s)
        prev = s
    return out


def main():
    for dataset, label in (("non-square", "SHAPE_long_default"),
                           ("1024 aes", "SHAPE_long_1024aes")):
        timed = load(label)
        shapes = shape_sequence(dataset, need=len(timed) + 50)
        rows = classify(timed, shapes)
        print(f"=== {label} ({dataset}) — {len(rows)} timed steps ===")
        by = {}
        for s, t, k in rows:
            by.setdefault(k or "?", []).append(t)
        for k in ("repeat", "revisit", "first"):
            sel = by.get(k, [])
            if sel:
                print(f"  {k:8s} n={len(sel):3d} median {statistics.median(sel):.3f}s "
                      f"mean {statistics.mean(sel):.3f}s")
        rep, rev = by.get("repeat", []), by.get("revisit", [])
        if rep and rev:
            print(f"  revisit / repeat (median) = "
                  f"{statistics.median(rev) / statistics.median(rep):.2f}x")
            print(f"  revisit / repeat (mean)   = "
                  f"{statistics.mean(rev) / statistics.mean(rep):.2f}x")
        print()

    # The residual, by shape area, on the multi-shape dataset only.
    timed = load("SHAPE_long_default")
    shapes = shape_sequence("non-square", need=len(timed) + 50)
    rows = [(s, t, k) for s, t, k in classify(timed, shapes) if s and k in ("repeat", "revisit")]
    buckets = {}
    for s, t, k in rows:
        buckets.setdefault(area(s) // 1000 * 1000, []).append(t)
    print("non-square, steady-state steps only (repeat+revisit), by latent area:")
    for b in sorted(buckets):
        sel = buckets[b]
        print(f"  area {b:5d}-{b + 999:5d}: n={len(sel):4d} median {statistics.median(sel):.3f}s")
    print()
    print("if the penalty were per-shape and constant, these medians would be flat;")
    print("if it were compute, the larger shapes would be slower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
