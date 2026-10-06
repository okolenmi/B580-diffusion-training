#!/usr/bin/env python3
"""Identify which thread burns the CPU during a first sighting, and how much.

The measured signature is: during a slow step, CPU time / wall time is 1.00 --
one thread at 100% while the GPU sits at 25-30%. "One thread" is the whole
diagnosis, but it does not say *which*, and the answer decides which lever
applies:

  * the main Python thread      -> the cost is under our own code, and JIT or
                                   kernel generation is being driven inline
  * a UR / SYCL / Level-Zero worker -> the runtime is compiling off-thread and
                                   the wait is ours but the work is not
  * an OpenMP or oneDNN JIT worker -> oneDNN's own JIT, parallelisable with
                                   more threads, and the single-threaded
                                   signature would mean it is NOT using them

Thread *names* come from /proc/<pid>/task/<tid>/comm and CPU from utime+stime
in the same file's stat, so this needs no profiler installed and no ptrace.

Usage:  python3 thread_watch.py <pid> [--interval 0.2] [--out csv]
Sampled until the process exits or Ctrl-C. Prints the top consumers at the end,
and a per-second CSV so a specific slow window can be correlated with the
run's own steps.jsonl (same clock, both wall seconds).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

CLK = os.sysconf("SC_CLK_TCK")


def threads(pid: int) -> dict[int, tuple[str, float]]:
    """tid -> (comm, cpu_seconds). A thread that exits mid-sample is simply
    absent from one sample, which is why this returns a fresh dict each time
    rather than accumulating."""
    out: dict[int, tuple[str, float]] = {}
    base = Path(f"/proc/{pid}/task")
    try:
        tids = os.listdir(base)
    except OSError:
        return out
    for tid in tids:
        try:
            comm = (base / tid / "comm").read_text().strip()
            stat = (base / tid / "stat").read_text()
            # comm may contain spaces and parentheses; fields after the last
            # ')' are stable, and utime/stime are fields 14/15 overall.
            tail = stat[stat.rindex(")") + 1:].split()
            utime, stime = int(tail[11]), int(tail[12])
            out[int(tid)] = (comm, (utime + stime) / CLK)
        except (OSError, ValueError, IndexError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pid", type=int)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--out")
    args = ap.parse_args()

    prev = threads(args.pid)
    prev_t = time.time()
    # The sample before `prev`, kept only so the busiest-thread ranking can
    # compare each thread against its own previous reading rather than against
    # its lifetime total.
    prev_prev: dict[int, tuple[str, float]] = {}
    # name -> seconds of CPU accumulated since the first sample
    totals: dict[str, float] = defaultdict(float)
    # name -> number of samples where this thread burned the most CPU *in that
    # sample*. Per-sample delta, not cumulative: a thread that has done the most
    # total work is the cumulative leader in every single sample, so ranking by
    # it reports the busiest thread as 99-100% of samples no matter what is
    # actually happening. That bug made an early version of this report "one
    # thread, 99% of samples" for a run that was really using 1.00 cores evenly
    # across two.
    busiest_count: dict[str, int] = defaultdict(int)
    writer = None
    if args.out:
        fh = open(args.out, "w", newline="", buffering=1)
        writer = csv.writer(fh)
        writer.writerow(["t", "comm", "cpu_delta", "cpu_total"])

    try:
        while True:
            time.sleep(args.interval)
            now = time.time()
            cur = threads(args.pid)
            if not cur:
                break
            window = now - prev_t
            for tid, (comm, cpu) in cur.items():
                before = prev.get(tid, (comm, cpu))[1]
                delta = max(0.0, cpu - before)
                totals[comm] += delta
                if writer:
                    writer.writerow([round(now, 3), comm, round(delta, 3),
                                     round(cpu, 3)])
            if cur:
                # Who burned the most CPU *this window*, which is the only
                # ranking that means anything per sample.
                top = max(cur.items(),
                          key=lambda kv: kv[1][1] - prev_prev.get(kv[0], (kv[1][0], kv[1][1]))[1])
                busiest_count[top[1][0]] += 1
            prev_prev, prev, prev_t = prev, cur, now
    except KeyboardInterrupt:
        pass

    print(f"{'thread name':32s} {'cpu s':>9s} {'% of samples busiest':>21s}")
    print("-" * 64)
    grand = sum(totals.values()) or 1.0
    n_samples = sum(busiest_count.values()) or 1
    for comm, secs in sorted(totals.items(), key=lambda kv: -kv[1])[:14]:
        print(f"{comm[:32]:32s} {secs:9.1f} {busiest_count[comm] / n_samples:20.0%}")
    print("-" * 64)
    print(f"total sampled CPU {grand:.1f} s across {len(totals)} distinct thread names")
    print()
    print("Reading it: a name that is the busiest in ~100% of samples is the one")
    print("to look at. Several names sharing it means the work is spread across a")
    print("pool, which is the opposite diagnosis -- a pool would mean the stall is")
    print("NOT single-threaded after all, and the cpu_frac=1.00 reading would be")
    print("summing a fraction of many threads rather than saturating one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
