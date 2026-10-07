#!/usr/bin/env python3
"""L2: read the shape-bucket sweep and answer both questions.

  (b) which multiple is fastest -- first-sighting seconds, steps/s, pad fraction
  (c) does bucketing cost quality -- holdout MSE on a fixed UNPADDED holdout

Everything is read from what the runs wrote, so this computes nothing that
could disagree with the measurement:

  summary.json  steps/s, peak reserved, holdout MSE + its digest, outcome
  steps.jsonl   one row per step, with `latent_shape` (so first sightings and
                revisits can be classified offline, exactly as
                `shapes diversity problem/analyze_steps.py` does)
  console.log   the build-time pad-fraction report, parsed back out

**The digest is checked before any comparison is printed.** Two runs that
scored different holdouts cannot be compared no matter what their MSEs say, so
a mismatch is reported as a broken comparison rather than as a result -- the
one place this script could otherwise produce a confident wrong answer.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "hw_validation"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def classify(rows: list[dict], skip: int) -> dict:
    """Split step durations into first sighting / revisit / repeat.

    Same definitions as `shapes diversity problem/analyze_steps.py`, which is
    where they were established: a *repeat* continues the previous step's
    shape, a *revisit* returns to a shape seen earlier, and a *first sighting*
    is one this process has never run.
    """
    seen, prev, prev_wall = set(), None, None
    buckets: dict[str, list[float]] = {"first": [], "revisit": [], "repeat": []}
    for row in rows:
        wall, shape, step = row.get("wall"), row.get("latent_shape"), row.get("step")
        if wall is None:
            continue
        if prev_wall is not None and shape is not None and step > skip:
            kind = ("repeat" if shape == prev
                    else ("first" if shape not in seen else "revisit"))
            buckets[kind].append(wall - prev_wall)
        if shape is not None:
            seen.add(shape)
            prev = shape
        prev_wall = wall
    return {"buckets": buckets,
            "distinct_shapes": len(seen),
            "first_sighting_seconds": sum(buckets["first"])}


def parse_pad_report(console: Path) -> dict:
    """Pull the pad-fraction line the dataset node printed at build."""
    if not console.exists():
        return {}
    text = console.read_text(errors="replace")
    out = {}
    m = re.search(r"pad fraction: median ([\d.]+)%, mean ([\d.]+)%, "
                  r"p90 ([\d.]+)%, max ([\d.]+)%; (\d+)/(\d+) sample", text)
    if m:
        out.update(pad_median=float(m.group(1)) / 100,
                   pad_mean=float(m.group(2)) / 100,
                   pad_p90=float(m.group(3)) / 100,
                   pad_max=float(m.group(4)) / 100,
                   padded_samples=int(m.group(5)),
                   samples=int(m.group(6)))
    m = re.search(r"shape bucketing x(\d+): (\d+) sample\(s\), (\d+) shape\(s\) "
                  r"in -> (\d+) bucket", text)
    if m:
        out.update(multiple=int(m.group(1)), shapes_in=int(m.group(3)),
                   shapes_out=int(m.group(4)))
    m = re.search(r"latent pixels x([\d.]+)", text)
    if m:
        out["latent_pixel_factor"] = float(m.group(1))
    return out


def load_run(label: str, skip: int) -> dict | None:
    d = RUNS / label
    summary_path = d / "summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if summary.get("outcome") != "ok":
        return {"label": label, "outcome": summary.get("outcome"),
                "config": summary.get("config", {})}
    rows = load_jsonl(d / "steps.jsonl")
    agg = summary.get("run_aggregates", {})
    stats = classify(rows, skip)
    out = {
        "label": label,
        "outcome": "ok",
        "config": summary.get("config", {}),
        "steps_recorded": agg.get("steps_recorded"),
        "steps_per_sec": agg.get("steps_per_sec_steady"),
        "peak_reserved_mb": (agg.get("per_step_peak_reserved_mb") or {}).get("max"),
        "distinct_shapes": stats["distinct_shapes"],
        "first_sighting_seconds": round(stats["first_sighting_seconds"], 1),
        "holdout": summary.get("holdout"),
        "pad": parse_pad_report(d / "console.log"),
    }
    for kind, values in stats["buckets"].items():
        out[f"{kind}_median_s"] = (round(statistics.median(values), 3)
                                   if values else None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", default="L2_x0,L2_x0_b,L2_x16,L2_x24,L2_x32")
    ap.add_argument("--skip", type=int, default=3,
                    help="ignore the first N steps (model load, first-touch)")
    args = ap.parse_args(argv)

    runs = [r for r in (load_run(l.strip(), args.skip)
                        for l in args.labels.split(",")) if r]
    if not runs:
        print(f"no runs found under {RUNS}")
        return 1
    bad = [r for r in runs if r["outcome"] != "ok"]
    if bad:
        for r in bad:
            print(f"SKIPPED {r['label']}: outcome={r['outcome']}")
        runs = [r for r in runs if r["outcome"] == "ok"]

    # -- (c) the holdout, and the digest check that makes it a comparison ----
    digests = {r["label"]: (r["holdout"] or {}).get("digest") for r in runs}
    usable = [r for r in runs if (r["holdout"] or {}).get("mse_mean") is not None]
    digests = {r["label"]: (r["holdout"] or {}).get("digest") for r in usable}
    distinct = set(d for d in digests.values() if d)

    print("=" * 78)
    print("(c) QUALITY: loss on a fixed UNPADDED holdout, same seed")
    print("=" * 78)
    if not usable:
        print("no run recorded a holdout")
    elif len(distinct) > 1:
        print(f"REFUSING TO COMPARE: {len(distinct)} different holdout digests "
              f"across the runs.")
        for label, digest in digests.items():
            print(f"    {label}: {digest}")
        print("  the arms scored different inputs, so their MSEs are not a "
              "comparison")
    else:
        for r in sorted(usable, key=lambda r: r["holdout"]["mse_mean"]):
            h = r["holdout"]
            print(f"  {r['label']:<10} mse {h['mse_mean']:.6f}  "
                  f"(min {h['mse_min']:.6f}, max {h['mse_max']:.6f}, "
                  f"{h['batches']} batches)")
        best = min(usable, key=lambda r: r["holdout"]["mse_mean"])
        print(f"  digest {next(iter(distinct))} (identical across all arms)")

        # The noise control: two runs of the SAME config. Any difference
        # between arms has to be read against this, not against zero.
        controls = [r for r in usable if r["label"].endswith("_b")]
        base = [r for r in usable if r["label"] == "L2_x0"]
        if controls and base:
            delta = abs(controls[0]["holdout"]["mse_mean"]
                        - base[0]["holdout"]["mse_mean"])
            print(f"\n  noise control (same config, different seed): "
                  f"|{delta:.6f}| difference")
            print(f"  {'a difference below this is not evidence of anything':>0}"
                  if delta else "  (control matched exactly)")
            for r in usable:
                if r["label"] in ("L2_x0", "L2_x0_b"):
                    continue
                gap = r["holdout"]["mse_mean"] - base[0]["holdout"]["mse_mean"]
                verdict = ("within noise" if abs(gap) <= delta
                           else ("WORSE" if gap > 0 else "better"))
                print(f"  {r['label']:<10} vs L2_x0: {gap:+.6f}  "
                      f"({gap / base[0]['holdout']['mse_mean'] * 100:+.2f}%) "
                      f"-> {verdict}")
        else:
            print(f"  best: {best['label']}")

    # -- (b) throughput ------------------------------------------------------
    print()
    print("=" * 78)
    print("(b) THROUGHPUT and cost")
    print("=" * 78)
    print(f"  {'run':<10}{'x':>4}{'shapes':>8}{'buckets':>9}"
          f"{'first sight':>13}{'steps/s':>9}{'peak MB':>9}"
          f"{'pad mean':>10}{'pad max':>9}{'latent x':>10}")
    for r in sorted(usable, key=lambda r: (r["config"].get(
            "shape_bucket_multiple") or 0, r["label"])):
        pad = r["pad"]
        print(f"  {r['label']:<10}"
              f"{r['config'].get('shape_bucket_multiple') or 0:>4}"
              f"{r['distinct_shapes']:>8}"
              f"{pad.get('shapes_out', '-'):>9}"
              f"{r['first_sighting_seconds']:>12.1f}s"
              f"{r['steps_per_sec'] or 0:>9.3f}"
              f"{r['peak_reserved_mb'] or 0:>9.0f}"
              f"{pad.get('pad_mean', 0) * 100:>9.1f}%"
              f"{pad.get('pad_max', 0) * 100:>8.1f}%"
              f"{pad.get('latent_pixel_factor', 1):>10.3f}")
    print()
    print("  first-sighting seconds is the one-time compile cost this feature "
          "exists to remove;")
    print("  steps/s is what remains after it. pad mean/max is the permanent "
          "cost, per sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
