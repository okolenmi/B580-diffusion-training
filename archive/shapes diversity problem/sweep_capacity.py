"""Smallest sufficient primitive-cache capacity, and what it costs in host RAM.

Two open questions, one sweep. Only 1024 (oneDNN's default, measured to cost
4.45x on revisits) and 65536 (measured to work) have been run, so the landed
value may be far larger than it needs to be -- and it is *host* memory, which
cannot OOM the card but is not free on a small-memory host.

Each capacity runs as its OWN process, because the oneDNN primitive cache
lives inside one and cannot be resized within it. Runs are sequential: they
share the card, and two at once caused DEVICE_LOST earlier.

150 steps rather than 300, which is the point where this becomes cheap rather
than minimal: `non-square` has 44 distinct shapes clumped at mean 2.16, so
first sightings end around step 44 and the remaining ~100 steps are revisits.
That is far more revisits than needed to see whether revisits are fast, and it
halves the cost per point. The 300-step runs in results/ are the reference for
the headline numbers; this sweep only has to rank capacities against each other.

Host RAM is the process's VmHWM -- peak resident set size, which is what the
kernel would refuse to grow past -- sampled while the run is alive. The
quantity that matters is the *difference* from the 1024 baseline, not any
absolute number, since the model's own resident footprint dominates both.

The five production variables from nodes/xpu_env.py are set here, because
that is the configuration a graph child actually gets and hw_validate.py does
not set them itself.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
PYTHON = "/home/okolenmi/comfy/venv/bin/python"
OUT = REPO / "runs" / "hw_validation"

PROD_ENV = {
    "SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD": "0",
    "SYCL_CACHE_IN_MEM": "1",
    "UR_L0_USE_RELAXED_ALLOCATION_LIMITS": "1",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
    "IGC_EnableDPEmulation": "1",
}


def run(capacity: int, steps: int) -> dict:
    label = f"CAP_{capacity}"
    log = OUT / f"{label}.console.log"
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **PROD_ENV, "ONEDNN_PRIMITIVE_CACHE_CAPACITY": str(capacity)}
    cmd = [PYTHON, "-u", "scripts/hw_validate.py", "main", "--label", label,
           "--dataset", "non-square", "--steps", str(steps), "--batch", "2",
           "--checkpoint", "div_4.safetensors"]
    started = time.time()
    peak_kb = 0
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=fh,
                                stderr=subprocess.STDOUT)
        status = Path(f"/proc/{proc.pid}/status")
        while proc.poll() is None:
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        peak_kb = max(peak_kb, int(line.split()[1]))
            except OSError:
                break
            time.sleep(0.5)
        rc = proc.returncode
    elapsed = time.time() - started

    result = dict(capacity=capacity, rc=rc, seconds=round(elapsed),
                  peak_rss_mb=peak_kb // 1024, steps_per_sec=None,
                  revisit_x=None, repeat=None, revisit=None, first=None)
    steps_path = OUT / label / "steps.jsonl"
    if rc != 0 or not steps_path.exists():
        return result

    rows = [json.loads(l) for l in steps_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step"])
    timed = [r for r in rows if r.get("dt_sec") and not r.get("covers_load")
             and r.get("latent_shape")]
    if not timed:
        return result
    seen, prev, kinds = set(), None, {"first": [], "repeat": [], "revisit": []}
    for r in timed:
        s, t = r["latent_shape"], r["dt_sec"]
        if s == prev:
            kinds["repeat"].append(t)
        elif s in seen:
            kinds["revisit"].append(t)
        else:
            kinds["first"].append(t)
        seen.add(s)
        prev = s
    med = lambda v: statistics.median(v) if v else None
    result.update(steps_per_sec=len(timed) / sum(r["dt_sec"] for r in timed),
                  repeat=med(kinds["repeat"]), revisit=med(kinds["revisit"]),
                  first=med(kinds["first"]), n_first=len(kinds["first"]),
                  n_revisit=len(kinds["revisit"]))
    if result["repeat"]:
        result["revisit_x"] = round(result["revisit"] / result["repeat"], 2)
    return result


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    caps = [int(a) for a in sys.argv[2:]] or [1024, 2048, 4096, 8192, 16384, 32768]
    print(f"sweep: {caps} at {steps} steps, production env, sequential")
    print(f"{'capacity':>9} {'steps/s':>8} {'revisit':>8} {'xsteady':>8} "
          f"{'peak RSS MB':>12} {'dRSS vs 1024':>14} {'run s':>7}")
    print("-" * 74)
    results = []
    baseline_rss = None
    for cap in caps:
        r = run(cap, steps)
        results.append(r)
        if cap == caps[0]:
            baseline_rss = r["peak_rss_mb"]
        delta = (r["peak_rss_mb"] - baseline_rss
                 if baseline_rss is not None and r["peak_rss_mb"] else None)
        if r["rc"] != 0:
            print(f"{cap:>9} FAILED rc={r['rc']} -- see {OUT}/CAP_{cap}.console.log")
            continue
        print(f"{cap:>9} {r['steps_per_sec']:>8.3f} {r['revisit']:>7.3f}s "
              f"{r['revisit_x']:>7.2f}x {r['peak_rss_mb']:>12} "
              f"{(f'+{delta}' if delta is not None and delta > 0 else (str(delta) if delta is not None else 'n/a')):>14} "
              f"{r['seconds']:>7}")
        (OUT.parent / "hw_validation").mkdir(exist_ok=True)
        Path(f"/tmp/cap_sweep_{cap}.json").write_text(json.dumps(r))

    print()
    ok = [r for r in results if r.get("revisit_x") is not None]
    good = [r for r in ok if r["revisit_x"] < 1.25]
    bad = [r for r in ok if r["revisit_x"] >= 1.25]
    if len(ok) < 2:
        # A one-point sweep cannot bracket anything: its own capacity would
        # come back as "the smallest that works", which is true only because
        # nothing smaller was tried. Said plainly rather than printed as a
        # finding -- an earlier version of this script did print the
        # single-point answer, which read as a result.
        print("\nonly one capacity ran, so nothing is bracketed; a 'smallest "
              "sufficient' claim needs the failing point and the working point "
              "in the same sweep.")
        Path("/tmp/cap_sweep_all.json").write_text(json.dumps(results, indent=1))
        return 0
    if good:
        smallest = min(r["capacity"] for r in good)
        print(f"smallest capacity measured that holds revisit/steady under 1.25: {smallest}")
        if bad:
            biggest_bad = max(r["capacity"] for r in bad)
            print(f"largest capacity measured that does NOT: {biggest_bad}")
            print(f"so the working value is bracketed by ({biggest_bad}, {smallest}]")
        else:
            print(f"no capacity in this sweep failed, so the bracket's lower end "
                  f"is below {min(r['capacity'] for r in ok)}")
        landed = int(os.environ.get("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "65536"))
        print(f"currently landed: {landed} = {landed / smallest:.0f}x the smallest "
              f"measured sufficient")
    else:
        print("no usable results")
    Path("/tmp/cap_sweep_all.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
