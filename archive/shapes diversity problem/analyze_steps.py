#!/usr/bin/env python3
"""Classify the steps of a REAL training run (scripts/hw_validate.py output) as warm repeats,
first sightings or revisits of a latent shape, and say whether the cost is one-time or recurring.

Inputs (all in one run directory):
  steps.jsonl   one row per step with 'step' and 'wall' (unix seconds)   [written by hw_validate]
  shapes.jsonl  one row per step: {"step": n, "shape": "HxW"}            [written by the logging patch in the plan]
  cache_watch.csv (optional)  from watch_cache.py, same clock as 'wall'

    python3 analyze_steps.py runs/hw/R0_baseline [--skip 3]

Definitions: a *repeat* is the same shape as the previous step (always warm: the steady state);
a *first sighting* is a shape not yet seen in this process; a *revisit* is a shape seen earlier but
not in the previous step. Revisits are the discriminator: fast => the cost was one-time;
slow => nothing is retained between visits (the project's observation: same speed at step 1000).
"""
from __future__ import annotations
import argparse, csv, json, statistics, sys
from pathlib import Path


def load_jsonl(path: Path):
    rows = []
    for ln in path.read_text().splitlines():
        ln = ln.strip()
        if ln:
            try: rows.append(json.loads(ln))
            except json.JSONDecodeError: pass
    return rows


def classify(steps, shapes, skip):
    by_step = {r["step"]: r["shape"] for r in shapes}
    out, seen, prev, prev_wall = [], set(), None, None
    for r in steps:
        n, wall = r["step"], r["wall"]
        shape = by_step.get(n)
        if prev_wall is not None and shape is not None and n > skip:
            kind = "repeat" if shape == prev else ("first" if shape not in seen else "revisit")
            out.append(dict(step=n, shape=shape, dt=wall - prev_wall, kind=kind, t0=prev_wall, t1=wall))
        if shape is not None:
            seen.add(shape); prev = shape
        prev_wall = wall
    return out


def cache_growth(rows, name, t0, t1):
    def at(t):
        best = None
        for r in rows:
            if r["name"] == name and r["t"] <= t: best = r
        return best
    a, b = at(t0), at(t1)
    if a is None or b is None: return None
    return b["files"] - a["files"], b["bytes"] - a["bytes"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("--skip", type=int, default=3, help="ignore the first N steps (model load, one-off init)")
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    d = Path(args.run_dir)
    steps = load_jsonl(d / "steps.jsonl")
    shapes = load_jsonl(d / "shapes.jsonl") if (d / "shapes.jsonl").exists() else []
    if not steps or not shapes:
        print(f"need steps.jsonl and shapes.jsonl in {d} (found {len(steps)} step rows, {len(shapes)} shape rows)."); return 1
    rows = classify(steps, shapes, args.skip)
    by = {k: [r for r in rows if r["kind"] == k] for k in ("repeat", "first", "revisit")}
    med = lambda sel: statistics.median(r["dt"] for r in sel) if sel else float("nan")
    print(f"{len(rows)} classified steps from {len(steps)} (skipped first {args.skip}); distinct shapes {len({r['shape'] for r in rows})}")
    for k in ("repeat", "first", "revisit"):
        print(f"  {k:8s} n={len(by[k]):4d}  median {med(by[k]):7.3f} s")
    if not by["repeat"] or not by["revisit"]:
        print("not enough repeats/revisits to decide (need a longer run or a dataset with consecutive same-shape batches)"); return 1
    steady = med(by["repeat"]); rev_x = med(by["revisit"]) / steady
    slow = [r for r in by["revisit"] if r["dt"] > 1.5 * steady]
    print(f"  steady state {steady:.3f} s; revisit median = {rev_x:.2f}x steady; {len(slow)}/{len(by['revisit'])} revisits >1.5x steady")
    half = len(rows) // 2
    late_rev = [r for r in rows[half:] if r["kind"] == "revisit"]
    if late_rev:
        print(f"  second half of the run: revisit median {med(late_rev) / steady:.2f}x steady ({len(late_rev)} revisits)")
    total = sum(r["dt"] for r in rows); lost = sum(max(0.0, r["dt"] - steady) for r in rows)
    print(f"  time above steady state: {lost:.0f} s of {total:.0f} s ({lost / total:.0%}) -- the share a fix could win back")
    cw = d / "cache_watch.csv"
    if cw.exists():
        cache = [dict(t=float(r["t"]), name=r["name"], files=int(r["files"]), bytes=int(r["bytes"])) for r in csv.DictReader(cw.open())]
        for name in sorted({r["name"] for r in cache}):
            g = [(r, cache_growth(cache, name, r["t0"], r["t1"])) for r in rows]
            g = [(r, x) for r, x in g if x is not None]
            if not g or not any(x[0] for _, x in g):
                print(f"  cache '{name}': no growth during any step"); continue
            slow_all = [(r, x) for r, x in g if r["dt"] > 1.5 * steady]
            grew = [1 for r, x in slow_all if x[0] > 0]
            print(f"  cache '{name}': grew during {sum(1 for _, x in g if x[0] > 0)} steps; {len(grew)}/{len(slow_all)} slow steps wrote to it")
    print()
    if slow and len(slow) / len(by["revisit"]) >= 0.3 or rev_x >= 1.3:
        print("VERDICT: RECURRING. A shape that was seen before is still slow when it returns: nothing is retained between visits.")
    elif len(by["first"]) and med(by["first"]) > 1.5 * steady:
        print("VERDICT: ONE-TIME. Only first sightings are slow.")
    else:
        print("VERDICT: no shape-dependent cost visible in this run.")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
