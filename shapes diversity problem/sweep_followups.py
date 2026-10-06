"""Three follow-ups to the capacity sweep, in one place because they share the card.

1. **Is capacity a ceiling or an allocation?** Peak RSS was byte-identical at
   every capacity from 1024 to 65536, which says unused capacity costs
   nothing. That is load-bearing for the design -- it is why a generous
   default is free and the adaptive formula only has to cover the cases a
   default misses. It is inferred from a 32x range, so this pushes it to
   256x (262144) to see whether RSS finally moves. If RSS is still flat,
   capacity is a limit; if it jumps, a generous default is NOT free and the
   formula has to be tight.

2. **How many primitives per shape?** The sweep bracketed the requirement for
   44 shapes to (1024, 2048], i.e. between 23 and 46 primitives per shape --
   too wide to size a cache for another dataset. Two points inside that
   bracket (1280, 1536) narrow it by roughly 4x, and the answer is what turns
   "pick a big number" into a rule that can be checked against a dataset's
   actual shape count.

3. Nothing else: the persistent-cache question is settled separately at 300
   steps, because it needs a longer run to be a fair test and mixing it in
   here would make the arm count confusing.

Sequential, one process per capacity, production env. Each run is a separate
process because the primitive cache cannot be resized within one.
"""

import json
import os
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
    label = f"CAP2_{capacity}"
    OUT.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **PROD_ENV, "ONEDNN_PRIMITIVE_CACHE_CAPACITY": str(capacity)}
    cmd = [PYTHON, "-u", "scripts/hw_validate.py", "main", "--label", label,
           "--dataset", "non-square", "--steps", str(steps), "--batch", "2",
           "--checkpoint", "div_4.safetensors"]
    t0 = time.time()
    peak_kb = 0
    with open(OUT / f"{label}.console.log", "w") as fh:
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
    res = dict(capacity=capacity, rc=rc, seconds=round(time.time() - t0),
               peak_rss_mb=peak_kb // 1024, steps_per_sec=None, revisit_x=None,
               revisit=None, repeat=None, n_shapes=None, n_revisit=None)
    p = OUT / label / "steps.jsonl"
    if rc or not p.exists():
        return res
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step"])
    timed = [r for r in rows if r.get("dt_sec") and not r.get("covers_load")
             and r.get("latent_shape")]
    seen, prev, rep, rev = set(), None, [], []
    for r in timed:
        s, t = r["latent_shape"], r["dt_sec"]
        if s == prev:
            rep.append(t)
        elif s in seen:
            rev.append(t)
        seen.add(s)
        prev = s
    med = lambda v: statistics.median(v) if v else None
    res.update(steps_per_sec=len(timed) / sum(r["dt_sec"] for r in timed),
               repeat=med(rep), revisit=med(rev), n_shapes=len(seen),
               n_revisit=len(rev))
    if res["repeat"]:
        res["revisit_x"] = round(res["revisit"] / res["repeat"], 2)
    return res


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    caps = [int(a) for a in sys.argv[2:]] or [1280, 1536, 262144]
    print(f"follow-ups at {steps} steps: {caps}")
    print(f"{'capacity':>9} {'shapes':>7} {'revisit':>8} {'xsteady':>8} "
          f"{'steps/s':>8} {'peak RSS MB':>12} {'run s':>7}")
    print("-" * 68)
    out = []
    for cap in caps:
        r = run(cap, steps)
        out.append(r)
        Path(f"/tmp/cap2_{cap}.json").write_text(json.dumps(r))
        if r["rc"]:
            print(f"{cap:>9} FAILED rc={r['rc']}")
            continue
        print(f"{cap:>9} {r['n_shapes']:>7} {r['revisit']:>7.3f}s "
              f"{r['revisit_x']:>7.2f}x {r['steps_per_sec']:>8.3f} "
              f"{r['peak_rss_mb']:>12} {r['seconds']:>7}")

    ok = [r for r in out if r.get("revisit_x") is not None]
    if ok:
        shapes = ok[0]["n_shapes"]
        bad = [r for r in ok if r["revisit_x"] >= 1.25]
        good = [r for r in ok if r["revisit_x"] < 1.25]
        print()
        if bad and good:
            hi_bad = min(r["capacity"] for r in bad)
            lo_good = min(r["capacity"] for r in good)
            print(f"{shapes} shapes need a capacity in ({hi_bad}, {lo_good}]")
            print(f"  -> {hi_bad / shapes:.1f} to {lo_good / shapes:.1f} "
                  f"primitives per shape")
        else:
            print(f"{shapes} shapes: all measured capacities "
                  f"{'failed' if bad else 'succeeded'}")
        rss = {r["peak_rss_mb"] for r in ok}
        print(f"  peak RSS across these runs: {sorted(rss)} MB")
        print("  (flat => capacity is a ceiling, so a generous default is free; "
              "a jump => unused capacity allocates and the formula must be tight)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
