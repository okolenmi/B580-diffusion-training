"""How much of the first-sighting cost survives across processes?

Every graph run is a fresh child process (MEM-05), so the 44 first sightings
are repaid in full on every run: 3.85 s x 44 = ~170 s. The primitive cache fix
(ONEDNN_PRIMITIVE_CACHE_CAPACITY) does nothing about that -- it stops
*revisits* paying, and first sightings still pay because they genuinely are new
code. Measured with the fix landed: repeat 0.93s, revisit 0.93s, first 3.85s.

A persistent on-disk kernel cache (`SYCL_CACHE_PERSISTENT=1` +
`SYCL_CACHE_DIR`) is the remaining lever, and it is a different layer again:
this persists SYCL's SPIR-V kernel binaries, not oneDNN primitives. So it may
remove part, all, or none of the first-sighting cost. That is the question.

Three runs, all with the primitive-cache fix in place so the revisit cost is
already out of the way and what remains is purely first-sighting:

  1. baseline       no persistent cache -- the cost every run pays today
  2. cold           persistent cache enabled, directory empty; pays full cost
     and populates the cache
  3. warm           same directory, second process; whatever it saves is what
     a repeat run of the same shapes actually gets

(2) and (3) are the pair that matters. If (3)'s first-sighting median drops
toward the repeat median, the cache is worth enabling and the ~170 s is a
once-per-machine cost rather than once-per-run.

The cache directory is deliberately an isolated path under /tmp rather than
~/.cache/sycl_kernels: this measures the cache, and writing to the user's real
cache would silently make every later measurement in this repo faster for
reasons that have nothing to do with what is being tested.
"""

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
PYTHON = "/home/okolenmi/comfy/venv/bin/python"
OUT = REPO / "runs" / "hw_validation"
CACHE_DIR = Path("/tmp/opencode/sycl_kernels_persist_test")

PROD_ENV = {
    "SYCL_IN_MEM_CACHE_EVICTION_THRESHOLD": "0",
    "SYCL_CACHE_IN_MEM": "1",
    "UR_L0_USE_RELAXED_ALLOCATION_LIMITS": "1",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS": "1",
    "IGC_EnableDPEmulation": "1",
    "ONEDNN_PRIMITIVE_CACHE_CAPACITY": "65536",
}


def dir_stats(path: Path) -> tuple[int, int]:
    files = size = 0
    for root, _dirs, names in os.walk(path):
        for n in names:
            try:
                files += 1
                size += (Path(root) / n).stat().st_size
            except OSError:
                pass
    return files, size


def run(label: str, steps: int, persistent: bool) -> dict:
    env = {**os.environ, **PROD_ENV}
    if persistent:
        env["SYCL_CACHE_PERSISTENT"] = "1"
        env["SYCL_CACHE_DIR"] = str(CACHE_DIR)
    else:
        env.pop("SYCL_CACHE_PERSISTENT", None)
        env.pop("SYCL_CACHE_DIR", None)
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, "-u", "scripts/hw_validate.py", "main", "--label", label,
           "--dataset", "non-square", "--steps", str(steps), "--batch", "2",
           "--checkpoint", "div_4.safetensors"]
    t0 = time.time()
    with open(OUT / f"{label}.console.log", "w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=fh,
                              stderr=subprocess.STDOUT, timeout=3600)
    out = dict(label=label, rc=proc.returncode, seconds=round(time.time() - t0),
               persistent=persistent, steps_per_sec=None, revisit_x=None,
               first=None, repeat=None, revisit=None)
    steps_path = OUT / label / "steps.jsonl"
    if proc.returncode or not steps_path.exists():
        return out
    rows = [json.loads(l) for l in steps_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["step"])
    timed = [r for r in rows if r.get("dt_sec") and not r.get("covers_load")
             and r.get("latent_shape")]
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
    out.update(steps_per_sec=len(timed) / sum(r["dt_sec"] for r in timed),
               repeat=med(kinds["repeat"]), revisit=med(kinds["revisit"]),
               first=med(kinds["first"]),
               n_first=len(kinds["first"]), n_revisit=len(kinds["revisit"]))
    if out["repeat"]:
        out["revisit_x"] = round(out["revisit"] / out["repeat"], 2)
    return out


def main() -> int:
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    # Start from nothing, so "cold" is genuinely cold.
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    print(f"persistent-cache test, {steps} steps each, primitive fix in place")
    print(f"isolated cache dir: {CACHE_DIR} (not the user's real one)")
    print()

    results = []
    for label, persistent in (("PERSIST_off", False), ("PERSIST_cold", True),
                              ("PERSIST_warm", True)):
        r = run(label, steps, persistent)
        results.append(r)
        files, size = dir_stats(CACHE_DIR) if CACHE_DIR.exists() else (0, 0)
        if r["rc"] != 0:
            print(f"{label:14s} FAILED rc={r['rc']} -- see "
                  f"{OUT}/{label}.console.log")
            continue
        print(f"{label:14s} first={r['first']:.3f}s repeat={r['repeat']:.3f}s "
              f"revisit={r['revisit']:.3f}s  {r['steps_per_sec']:.3f} steps/s"
              f"   cache now {files} files / {size / 2**20:.0f} MB")

    ok = [r for r in results if r.get("first")]
    if len(ok) == 3:
        off, cold, warm = ok
        print()
        print(f"first-sighting cost, no persistent cache : {off['first']:.3f} s "
              f"x {off['n_first']} = {off['first'] * off['n_first']:.0f} s per run")
        print(f"first-sighting cost, cache cold           : {cold['first']:.3f} s "
              f"x {cold['n_first']} = {cold['first'] * cold['n_first']:.0f} s")
        print(f"first-sighting cost, cache warm           : {warm['first']:.3f} s "
              f"x {warm['n_first']} = {warm['first'] * warm['n_first']:.0f} s")
        saved = cold["first"] * cold["n_first"] - warm["first"] * warm["n_first"]
        print()
        print(f"a warm cache saves {saved:.0f} s per run ({saved / max(1, cold['first'] * cold['n_first']):.0%})")
        print(f"throughput: {cold['steps_per_sec']:.3f} -> "
              f"{warm['steps_per_sec']:.3f} steps/s "
              f"({warm['steps_per_sec'] / cold['steps_per_sec']:.2f}x)")
    Path("/tmp/persist_results.json").write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
