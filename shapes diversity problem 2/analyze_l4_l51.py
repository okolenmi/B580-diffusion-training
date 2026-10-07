#!/usr/bin/env python3
"""L4 and L5.1: batch size against step time, and what activation
checkpointing costs in a real run.

Reads what the sweeps wrote. Nothing is recomputed that could disagree with the
measurement -- steps/s and peak reserved come from summary.json, and the
interesting quantity (images per second) is derived from them arithmetically.

The batch table's whole point is the derived column. `steps/s` is flat from
batch 1 to 4, which looks like "nothing happened"; it is the *images* column
that quadruples while it stays flat, and that is the premise L4's structural
change rests on. A table that only showed steps/s would read as a null result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "hw_validation"


def load(label):
    p = RUNS / label / "summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    a = s.get("run_aggregates", {})
    cfg = s.get("config", {})
    peak = (a.get("per_step_peak_reserved_mb") or {}).get("max")
    drift = (a.get("reserved_mb_series") or {}).get("drift")
    return {"label": label, "outcome": s.get("outcome"),
            "batch": cfg.get("batch"), "steps": a.get("steps_recorded"),
            "steps_per_sec": a.get("steps_per_sec_steady"),
            "peak_mb": peak, "drift_mb": drift,
            "no_checkpoint": cfg.get("no_checkpoint", False)}


def sec_per_step(sps):
    return 1000.0 / sps if sps else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batch-labels",
                    default="L4_b1,L4_b2,L4_b4,L4_b8")
    ap.add_argument("--ckpt-labels",
                    default="L5_1_ckpt_on,L5_1_ckpt_off")
    args = ap.parse_args(argv)

    rows = [r for r in (load(l.strip())
                        for l in args.batch_labels.split(",")) if r]
    if not rows:
        print(f"no runs found under {RUNS}")
        return 1
    rows.sort(key=lambda r: r["batch"] or 0)

    print("=" * 84)
    print("L4: is step time flat as batch grows?  (150 steps, shape_bucket_multiple=32)")
    print("=" * 84)
    print(f"  {'run':<10}{'batch':>7}{'steps/s':>10}{'ms/step':>10}"
          f"{'images/s':>11}{'img/s vs b1':>13}{'peak MB':>10}{'drift MB':>10}")
    base = None
    for r in rows:
        if r["outcome"] != "ok" or not r["steps_per_sec"]:
            print(f"  {r['label']:<10}{r['batch']:>7}  -- outcome={r['outcome']}")
            continue
        images = r["steps_per_sec"] * r["batch"]
        if base is None:
            base = images
        print(f"  {r['label']:<10}{r['batch']:>7}{r['steps_per_sec']:>10.3f}"
              f"{sec_per_step(r['steps_per_sec']):>10.0f}{images:>11.2f}"
              f"{images / base:>12.2f}x{r['peak_mb'] or 0:>10.0f}"
              f"{r['drift_mb'] or 0:>10.0f}")

    ok = [r for r in rows if r["outcome"] == "ok" and r["steps_per_sec"]]
    if len(ok) >= 2:
        first, last = ok[0], ok[-1]
        span = [r for r in ok if (first["steps_per_sec"] * 0.95
                                  <= r["steps_per_sec"] <= first["steps_per_sec"] * 1.05)]
        if span:
            worst = max(span, key=lambda r: r["batch"])
            print()
            print(f"  step time is flat (within 5%) from batch {first['batch']} "
                  f"through batch {worst['batch']}:")
            print(f"    {first['steps_per_sec']:.3f} -> {worst['steps_per_sec']:.3f} "
                  f"steps/s while images/s goes "
                  f"{first['steps_per_sec'] * first['batch']:.2f} -> "
                  f"{worst['steps_per_sec'] * worst['batch']:.2f} "
                  f"({worst['steps_per_sec'] * worst['batch'] / (first['steps_per_sec'] * first['batch']):.2f}x)")
        grew = [r for r in ok if r["steps_per_sec"] < first["steps_per_sec"] * 0.95]
        if grew:
            g = grew[0]
            print(f"  it stops being flat at batch {g['batch']}: "
                  f"{g['steps_per_sec']:.3f} steps/s "
                  f"({sec_per_step(g['steps_per_sec']):.0f} ms/step), so images/s "
                  f"gains only {g['steps_per_sec'] * g['batch'] / (ok[-2]['steps_per_sec'] * ok[-2]['batch']) if len(ok) > 1 else 0:.2f}x "
                  f"over the previous batch for "
                  f"{(g['peak_mb'] or 0) - (ok[-2]['peak_mb'] or 0):+.0f} MB "
                  f"more reserved")

    print()
    print("=" * 84)
    print("L5.1: activation checkpointing, in a FULL run (150 steps, batch 2)")
    print("=" * 84)
    print(f"  {'run':<22}{'outcome':>10}{'steps/s':>10}{'ms/step':>10}"
          f"{'steps done':>12}{'peak MB':>10}")
    for label in (l.strip() for l in args.ckpt_labels.split(",")):
        r = load(label)
        if r is None:
            print(f"  {label:<22}  -- no run")
            continue
        sps = r["steps_per_sec"]
        print(f"  {label:<22}{r['outcome']:>10}"
              f"{sps:>10.3f}" if sps else f"  {label:<22}{r['outcome']:>10}{'-':>10}",
              end="")
        print(f"{sec_per_step(sps):>10.0f}{r['steps']:>12}{r['peak_mb'] or 0:>10.0f}"
              if sps else f"{'-':>10}{r['steps']:>12}{r['peak_mb'] or 0:>10.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
