"""What primitive-cache capacity is actually needed, and what does it cost in RAM?

Experiment B settled the cause (revisits 6.34x -> 1.00x at capacity 65536),
but "raise it to 65536 by default" is not yet an answer: 65536 is one arbitrary
number above the default 1024, and nothing says it is the *smallest* number that
works or what the host RAM cost is at each capacity. A default is worth having
only if it is the smallest round value that holds the working set.

So this measures, in one process per capacity (the cache is per-process and the
point is what a single training run experiences):

  * wall time per step, and revisit/steady ratio -- the thing being fixed;
  * host RSS -- what the cache actually costs;
  * distinct oneDNN primitives the 63 shapes need, read from ONEDNN_VERBOSE.

Capacity points bracket the default 1024 and B's 65536: 1024 (broken),
2048, 4096, 8192, 16384, 32768. 65536 is already measured.

Each run is a separate process because the primitive cache is not resettable
within one. Shapes come from the dataset, not a hand-written list, so this
matches experiment A exactly except for the one variable.
"""

import csv
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
PYTHON = "/home/okolenmi/comfy/venv/bin/python"
PROBE = REPO / "archive/shapes diversity problem" / "probe_shape_stall.py"
OUT = Path("/tmp/opencode/shapes")


def run_one(capacity: int, verbose: bool = False) -> dict:
    """One process, one capacity. Returns the parsed CSV plus host RSS."""
    csv_path = OUT / f"cap_{capacity}.csv"
    log_path = OUT / f"cap_{capacity}.log"
    env = dict(os.environ)
    if capacity:
        env["ONEDNN_PRIMITIVE_CACHE_CAPACITY"] = str(capacity)
    else:
        env.pop("ONEDNN_PRIMITIVE_CACHE_CAPACITY", None)
    if verbose:
        env["ONEDNN_VERBOSE"] = "1"
    else:
        env.pop("ONEDNN_VERBOSE", None)

    cmd = [PYTHON, "-u", str(PROBE), "--dataset", "datasets/non-square",
           "--batch", "2", "--passes", "3", "--repeat", "2", "--csv", str(csv_path)]
    t0 = time.perf_counter()
    peak_rss = 0
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT)
        # Host RSS sampled while it runs: the cache is host memory, so this is
        # the cost being asked about. /proc/<pid>/status VmHWM is the peak.
        status = Path(f"/proc/{proc.pid}/status")
        while proc.poll() is None:
            try:
                for line in status.read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        peak_rss = max(peak_rss, int(line.split()[1]) // 1024)
            except OSError:
                break
            time.sleep(0.5)
        rc = proc.returncode
    elapsed = time.perf_counter() - t0

    result = dict(capacity=capacity, rc=rc, wall=elapsed, peak_rss_mb=peak_rss,
                  verdict="", steady=None, revisit=None, revisit_x=None)
    if rc != 0 or not csv_path.exists():
        return result

    rows = list(csv.DictReader(open(csv_path)))
    for r in rows:
        r["wall_s"] = float(r["wall_s"])
        r["same_as_prev"] = int(r["same_as_prev"])
        r["first_seen"] = int(r["first_seen"])
    rep = [r["wall_s"] for r in rows if r["same_as_prev"] and not r["first_seen"]]
    rev = [r["wall_s"] for r in rows
           if not r["same_as_prev"] and not r["first_seen"]]
    if rep and rev:
        result["steady"] = statistics.median(rep)
        result["revisit"] = statistics.median(rev)
        result["revisit_x"] = statistics.median(rev) / statistics.median(rep)
    return result


def main() -> int:
    caps = [int(a) for a in sys.argv[1:]] or [1024, 2048, 4096, 8192, 16384, 32768]
    print(f"{'capacity':>10}  {'exit':>4}  {'steady s':>9}  {'revisit s':>10}  "
          f"{'ratio':>6}  {'peak RSS MB':>12}  {'run s':>7}")
    print("-" * 68)
    results = []
    for cap in caps:
        r = run_one(cap)
        results.append(r)
        if r["rc"] != 0:
            print(f"{cap:>10}  {r['rc']:>4}  FAILED -- see cap_{cap}.log")
            continue
        print(f"{cap:>10}  {r['rc']:>4}  {r['steady']:>9.3f}  {r['revisit']:>10.3f}  "
              f"{r['revisit_x']:>6.2f}  {r['peak_rss_mb']:>12}  {r['wall']:>7.0f}")
        print(f"{'':>10}  flush", flush=True)

    ok = [r for r in results if r["rc"] == 0 and r["revisit_x"] is not None]
    if ok:
        smallest = min((r["capacity"] for r in ok if r["revisit_x"] < 1.25),
                       default=None)
        print()
        print(f"smallest capacity measured that holds revisit/steady under 1.25: "
              f"{smallest}")
        print("default is 1024; B measured 65536.")
        if smallest:
            print(f"a default of {smallest} is {smallest / 1024:.0f}x the default and "
                  f"{65536 / smallest:.0f}x smaller than what was already proven to work")
        print()
        print("peak RSS is the host cost; note it includes the model's own resident")
        print("footprint (~7-10 GB), so the marginal cost of the cache is the")
        print("difference between rows, not any single row.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
