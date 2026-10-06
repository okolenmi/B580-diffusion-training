#!/usr/bin/env python3
"""Summarise oneDNN verbose output (ONEDNN_VERBOSE=1 or 2) per training step.

Input: the stderr/stdout capture of a run. Step boundaries come from marker
lines `PROBE_STEP,<pass>,<idx>,<rep>,<HxW>` (the probe prints them with --mark;
the trainer patch in the plan prints the same format as `PROBE_STEP,0,<step>,0,<HxW>`).

What it answers:
  * how many primitives were CREATED (cache_miss) vs reused (cache_hit) per step,
    and how many milliseconds went into creating them;
  * how many DISTINCT primitives one step needs, and therefore how many distinct
    primitives all shapes need: if (per-shape count x shapes) is far above the
    primitive-cache capacity (default 1024), the cache must thrash.

Formats handled (oneDNN changed them between releases):
  onednn_verbose,primitive,create:cache_miss,gpu,convolution,jit:ir,...,<ms>
  onednn_verbose,primitive,exec,gpu,matmul,...,<ms>
  onednn_verbose,create:cache_hit,gpu,...            (older, no 'primitive' token)
Lines that do not parse are counted, not guessed at.
"""
from __future__ import annotations
import argparse, collections, re, statistics, sys

KIND = re.compile(r"^(create(?::(?:cache_hit|cache_miss))?|exec|dispatch)$")


def parse_line(line: str):
    """-> (kind, engine, primitive, descriptor, ms) or None."""
    if not line.startswith("onednn_verbose,"):
        return None
    f = line.rstrip("\n").split(",")
    i = 1
    if i < len(f) and re.fullmatch(r"v\d+", f[i]):      # version token in newer releases
        i += 1
    if i < len(f) and f[i] == "primitive":
        i += 1
    if i >= len(f) or not KIND.match(f[i]):
        return None
    kind = f[i]
    engine = f[i + 1] if i + 1 < len(f) else ""
    prim = f[i + 2] if i + 2 < len(f) else ""
    try:
        ms = float(f[-1])
    except ValueError:
        ms = 0.0
    desc = ",".join(f[i + 1:-1])       # everything that identifies the primitive
    return kind, engine, prim, desc, ms


def segment(lines):
    """Yield (marker or None, [parsed lines]) per step; lines before the first marker get marker None."""
    cur, buf, unparsed = None, [], 0
    for ln in lines:
        if ln.startswith("PROBE_STEP,"):
            yield cur, buf, unparsed
            cur, buf, unparsed = ln.strip().split(",", 1)[1], [], 0
            continue
        p = parse_line(ln)
        if p:
            buf.append(p)
        elif ln.startswith("onednn_verbose"):
            unparsed += 1
    yield cur, buf, unparsed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--capacity", type=int, default=1024, help="primitive cache capacity in effect (ONEDNN_PRIMITIVE_CACHE_CAPACITY)")
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    steps = []
    all_distinct: set[str] = set()
    per_shape_distinct: dict[str, set[str]] = collections.defaultdict(set)
    unparsed_total = 0
    with open(args.log, errors="replace") as fh:
        for marker, parsed, unparsed in segment(fh):
            unparsed_total += unparsed
            if marker is None and not parsed:
                continue
            miss = [p for p in parsed if p[0] == "create:cache_miss" or p[0] == "create"]
            hit = [p for p in parsed if p[0] == "create:cache_hit"]
            execs = [p for p in parsed if p[0] == "exec"]
            shape = marker.split(",")[-1] if marker else "?"
            for p in parsed:
                # creation lines only: an exec line can describe the same primitive with a different string
                if p[0].startswith("create"):
                    all_distinct.add(p[3]); per_shape_distinct[shape].add(p[3])
            steps.append(dict(marker=marker or "(before first marker)", shape=shape, misses=len(miss), miss_ms=sum(p[4] for p in miss),
                              hits=len(hit), execs=len(execs), exec_ms=sum(p[4] for p in execs)))
    if not steps:
        print("no oneDNN verbose lines found. Was ONEDNN_VERBOSE set, and does this build print to stderr?")
        return 1
    n_miss = sum(s["misses"] for s in steps); n_hit = sum(s["hits"] for s in steps)
    print(f"{len(steps)} segments; primitives created (cache_miss) {n_miss}, reused (cache_hit) {n_hit}, "
          f"creation time {sum(s['miss_ms'] for s in steps) / 1000:.1f} s, unparsed verbose lines {unparsed_total}")
    with_miss = [s for s in steps if s["misses"]]
    print(f"steps that created at least one primitive: {len(with_miss)} of {len(steps)}")
    if with_miss:
        print(f"  median primitives created in such a step: {statistics.median(s['misses'] for s in with_miss):.0f}; "
              f"median creation time: {statistics.median(s['miss_ms'] for s in with_miss):.0f} ms")
    shapes = [s for s in per_shape_distinct if s != "?"]
    if shapes:
        per_shape = statistics.median(len(per_shape_distinct[s]) for s in shapes)
        need = per_shape * len(shapes)
        print(f"distinct primitives needed by one shape (median): {per_shape:.0f}; shapes seen: {len(shapes)}; "
              f"all shapes together (upper bound): {need:.0f}; total distinct seen: {len(all_distinct)}")
        print(f"cache capacity in effect: {args.capacity}")
        if need > args.capacity:
            print(f"  -> {need:.0f} > {args.capacity}: the primitive cache cannot hold every shape; revisits will re-create primitives. "
                  f"(ONEDNN_PRIMITIVE_CACHE_CAPACITY>={int(need * 1.2)} would hold them.)")
        else:
            print("  -> everything fits in the cache; revisit creation, if any, is not capacity eviction.")
    late = steps[len(steps) // 2:]
    if late:
        lm = sum(s["misses"] for s in late)
        print(f"second half of the run: {lm} primitive creations in {len(late)} segments"
              + ("  <- STILL creating late in the run: nothing is being retained" if lm > 0 else "  <- creation stopped: one-time cost"))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(steps[0])); w.writeheader(); w.writerows(steps)
        print("wrote", args.csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
