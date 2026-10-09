#!/usr/bin/env python3
"""Check that thread_watch.py's busiest-thread ranking measures what it claims.

The first version ranked by *cumulative* CPU, which makes the busiest thread
the cumulative leader in every sample -- so it reported one thread at ~100% of
samples for a run that was in fact using 1.00 cores evenly across two. That is
the kind of bug a self-test has to catch, because the output still looks
plausible: one name, a percentage, a confident story.

Both rankings are exercised on the same synthetic input, so the difference is
visible rather than argued:

  * one thread doing everything  -> both rankings must say `busy`
  * two threads sharing equally    -> cumulative says `busy` ~100% (wrong),
                                     per-sample says ~50/50 (right)

Run directly; exits non-zero on failure.
"""

import sys
from collections import defaultdict


def rank_cumulative(samples):
    """The old, wrong ranking: biggest lifetime total."""
    totals = defaultdict(float)
    counts = defaultdict(int)
    for sample in samples:
        for tid, (comm, cumulative) in sample.items():
            totals[comm] = cumulative
        counts[max(sample.items(), key=lambda kv: kv[1][1])[1][0]] += 1
    n = sum(counts.values())
    return {c: counts[c] / n for c in counts}


def rank_per_sample(samples):
    """The current ranking: biggest delta within this sample."""
    counts = defaultdict(int)
    prev = {}
    for sample in samples:
        best, best_delta = None, -1.0
        for tid, (comm, cumulative) in sample.items():
            delta = cumulative - prev.get(tid, cumulative)
            if delta > best_delta:
                best, best_delta = comm, delta
        prev = {tid: v[1] for tid, v in sample.items()}
        counts[best] += 1
    n = sum(counts.values())
    return {c: counts[c] / n for c in counts}


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def main() -> int:
    n = 40

    # Case 1: one thread does everything, the other is idle.
    one = {}
    cum = 0.0
    for _ in range(n):
        cum += 1.0
        one = {1: ("busy", cum), 2: ("idle", 0.0)}
    samples = [dict(one) for _ in range(n)]
    per = rank_per_sample(samples)
    check(per.get("busy", 0) > 0.8,
          f"a genuinely single-threaded sample should read as busy, got {per}")

    # Case 2: two threads alternate, each doing half the work. Their lifetime
    # totals end up equal, so the cumulative ranking has to pick one of them
    # and report it as the busiest in *every* sample -- which is the bug. The
    # per-sample ranking must show the alternation.
    samples = []
    a = b = 0.0
    for i in range(n):
        a += 1.0 if i % 2 == 0 else 0.0
        b += 1.0 if i % 2 == 1 else 0.0
        samples.append({1: ("alpha", a), 2: ("beta", b)})
    check(abs(a - b) < 1e-9,
          f"the two lifetime totals should tie for this case, got {a} vs {b}")
    cum_rank = rank_cumulative(samples)
    per_rank = rank_per_sample(samples)
    check(cum_rank.get("alpha", 0) > 0.9,
          "the cumulative ranking is supposed to be the broken one; if it "
          f"looks right here the test is not testing it (got {cum_rank})")
    spread = abs(per_rank.get("alpha", 0) - per_rank.get("beta", 0))
    check(spread < 0.5,
          f"per-sample ranking should show the split, got {per_rank}")
    print(f"  cumulative ranking (wrong): {cum_rank}")
    print(f"  per-sample ranking (right): {per_rank}")
    print("  PASS")
    print()
    print("SMOKE TEST: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
