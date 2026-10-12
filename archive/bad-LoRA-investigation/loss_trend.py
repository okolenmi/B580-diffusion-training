"""Is this run making progress? Segment the loss curve, don't eyeball it.

The problem this replaces. A raw per-step loss mixes every timestep, so at
batch 4 it swings 2-5x between neighbours -- which is why
`CAPTION_div4`'s trace could be read as "converged to 0.05 then diverged"
or "noisy but fine" and neither reading could be ruled out from the log
alone. An EMA smooths the jitter but still shows no *shape*, so a loss
that falls, plateaus, and climbs reads the same as one that falls
steadily. A 3-way timestep split does not help: it answers "which
timestep is hard", a different question, and its boundaries keep moving.

What this does instead, and why it is better than N fixed segments:

1. Fit ONE global trend to the whole run. Loss decays roughly
   exponentially, so the fit is on log(loss) -- straight there. This is
   the "overall trend" a line through the middle would give.

2. Split it into K pieces, and let the DATA choose the breakpoints by
   minimising total squared error. That is classic dynamic-programming
   least-squares segmentation: exact, not greedy, and it finds a break
   that a fixed grid walks straight past.

The reason this beats the alternative the maintainer described. Handing
the curve "draw 2 lines" or "draw 4 lines" fixes the breakpoints at
fractions of the run. A run that is flat for 80% and then collapses in the
last 5% gets one steep segment and 3 flat ones, which reads as "steady
progress" -- precisely the anomaly to catch. Free breakpoints put the
break where the data breaks.

3. Report each segment's slope in log space. Negative = improving,
   positive = getting worse. So "progress, then regression" is one line
   of output instead of a shape you have to notice.

Reading the verdict:
  * monotone negative slopes  -> converging, no anomalies
  * one or more positive slopes -> a regression segment; the step where
    it starts is printed
  * a late positive slope      -> the classic case above, and the fixed
    grid would have missed it

Usage:
    python -u archive/bad-LoRA-investigation/loss_trend.py runs/B_arm6.log
    python -u ... --segments 6 runs/*.log
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

#: a2_mine's progress line: "  [progress] step 900 loss=0.28235 ema=... shape=96x64"
STEP_RE = re.compile(
    r"\[progress\]\s+step\s+(\d+)\s+loss=([0-9.eE+-]+)"
    r"(?:\s+ema=([0-9.eE+-]+))?")


@dataclass
class Seg:
    """One fitted piece of the curve."""
    start: int
    end: int          # inclusive step index into the series
    slope: float      # per-step change in log(loss); negative = improving
    intercept: float
    r2: float

    @property
    def n(self) -> int:
        return self.end - self.start + 1

    def describe(self, steps: list[int]) -> str:
        word = "improving" if self.slope < 0 else (
            "FLAT" if self.slope == 0 else "REGRESSING")
        return (f"  steps {steps[self.start]:>6d}-{steps[self.end]:<6d} "
                f"n={self.n:<5d} slope={self.slope:+.6f}/step  "
                f"R2={self.r2:.4f}  {word}")


def read_series(path: str) -> tuple[list[int], list[float]]:
    """(steps, losses) from a steps JSONL if there is one, else the log.

    The JSONL is every step; the log only has the --progress lines. They
    are not interchangeable: a 1000-step run at --progress 100 gives ten
    points, and fitting a trend to ten noisy points measures the noise
    (R2=0.03 on arm6, which reads as "no signal" rather than "no
    progress"). So the dense source is preferred and its absence is
    reported rather than silently analysed anyway.
    """
    p = Path(path)
    if p.suffix == ".jsonl":
        steps, losses = [], []
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
                v = float(row["loss"])
            except (ValueError, KeyError, TypeError):
                continue
            if v > 0 and math.isfinite(v):
                steps.append(int(row["step"]))
                losses.append(v)
        return steps, losses
    steps, losses = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                v = float(m.group(2))
                if v > 0 and math.isfinite(v):   # log() needs this
                    steps.append(int(m.group(1)))
                    losses.append(v)
    return steps, losses


def _fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least squares y = slope*x + intercept, plus R2."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx else 0.0
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


def _sse(pre, i: int, j: int) -> float:
    """Sum of squared residuals for a straight-line fit over span [i, j).

    From prefix sums of x, y, xy, xx, yy, so every span costs O(1) and the
    DP over all spans stays O(n^2) overall rather than O(n^3).
    """
    sx, sy, sxy, sxx, syy = (pre[name][j] - pre[name][i] for name in
                             ("x", "y", "xy", "xx", "yy"))
    m = j - i
    if m < 2:
        return 0.0
    den = sxx - sx * sx / m
    if den <= 0:                      # all x identical: no slope to fit
        return syy - sy * sy / m
    return syy - sy * sy / m - (sxy - sx * sy / m) ** 2 / den


def segment(x: list[float], y: list[float], k: int,
            min_len: int = 5) -> list[tuple[int, int]]:
    """Exact least-squares split into <=k contiguous pieces.

    Classic DP: best[i][j] is the minimum SSE for x[0:j] using i pieces.
    O(k n^2), which is milliseconds at run lengths here and, unlike a
    greedy scan from the left, cannot be led astray by a long flat run
    early on.

    `min_len` is not cosmetic. Without it the optimum degenerates: a
    single noisy point always fits its own line perfectly (SSE 0), so the
    DP happily spends pieces isolating outliers and reports "no
    regression" on a curve that clearly has one. Every piece must hold at
    least this many points for its slope to mean anything.
    """
    n = len(x)
    if n == 0 or k <= 1:
        return [(0, n - 1)] if n else []
    min_len = max(2, min(min_len, max(1, n // max(k, 1))))
    k = min(k, max(1, n // min_len))
    return _solve(x, y, k, min_len)


def _solve(x: list[float], y: list[float], k: int, min_len: int) -> list[tuple[int, int]]:
    """The DP itself, once `segment` has clamped k and min_len."""
    n = len(x)

    # SSE of every contiguous span, via prefix sums so this stays O(n^2).
    pre = {"x": [0.0] * (n + 1), "y": [0.0] * (n + 1), "xy": [0.0] * (n + 1),
           "xx": [0.0] * (n + 1), "yy": [0.0] * (n + 1)}
    for i in range(n):
        pre["x"][i + 1] = pre["x"][i] + x[i]
        pre["y"][i + 1] = pre["y"][i] + y[i]
        pre["xy"][i + 1] = pre["xy"][i] + x[i] * y[i]
        pre["xx"][i + 1] = pre["xx"][i] + x[i] * x[i]
        pre["yy"][i + 1] = pre["yy"][i] + y[i] * y[i]

    INF = float("inf")
    best = [[INF] * (n + 1) for _ in range(k + 1)]
    back = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0.0
    for i in range(1, k + 1):
        lo = (i - 1) * min_len
        for j in range(i * min_len, n + 1):
            for m in range(lo, j - min_len + 1):
                if m == 0 and i > 1:
                    continue          # first piece must be non-empty
                if best[i - 1][m] == INF:
                    continue
                cost = best[i - 1][m] + _sse(pre, m, j)
                if cost < best[i][j]:
                    best[i][j] = cost
                    back[i][j] = m

    pieces: list[tuple[int, int]] = []
    j = n
    for i in range(k, 0, -1):
        if best[i][j] == INF:
            continue
        m = back[i][j]
        pieces.append((m, j - 1))
        j = m
        if j == 0:
            break
    pieces.reverse()
    return [p for p in pieces if p[1] >= p[0]] or ([(0, n - 1)] if n else [])


def analyse(steps: list[int], losses: list[float], k: int) -> None:
    x = [float(s) for s in steps]
    y = [math.log(v) for v in losses]          # log space: decay is linear

    print(f"\n{len(x)} points, steps {x[0]:.0f}-{x[-1]:.0f}")
    gslope, gint, gr2 = _fit(x, y)
    print(f"\noverall trend (one line, log space): "
          f"slope={gslope:+.6f}/step  R2={gr2:.4f}")
    print(f"  -> the run is "
          f"{'IMPROVING' if gslope < 0 else 'NOT improving'} overall; "
          f"a {math.exp(gslope * 100):.2f}x change per 100 steps")

    pieces = segment(x, y, k)
    print(f"\noptimal {len(pieces)}-segment fit (breakpoints chosen by the data):")
    segs = []
    for (a, b) in pieces:
        xs, ys = x[a:b + 1], y[a:b + 1]
        s, i0, r2 = _fit(xs, ys)
        segs.append(Seg(a, b, s, i0, r2))
    for sgm in segs:
        print(sgm.describe(steps))

    bad = [sgm for sgm in segs if sgm.slope > 0]
    print()
    if not bad:
        print("VERDICT: no regressing segment -- every piece improves.")
    else:
        print(f"VERDICT: {len(bad)} regressing segment(s):")
        for sgm in bad:
            print(f"  loss RISES from step {steps[sgm.start]} to "
                  f"{steps[sgm.end]} ({math.exp(sgm.slope * sgm.n):.2f}x over "
                  f"{sgm.n} steps)")
        print("  -> an EMA or a single trend line hides this; the breakpoints "
              "are where the data actually changed behaviour.")
    # A late regression is the case a fixed 2- or 4-way split misses.
    if bad and steps[bad[0].start] > 0.6 * x[-1]:
        print("  NOTE: it starts in the last 40% of the run -- a fixed-segment "
              "split at 1/2 and 1/4 would have averaged it away.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="run log(s) to analyse")
    ap.add_argument("--segments", type=int, default=4,
                    help="how many pieces the optimal split may use (default 4)")
    ap.add_argument("--min-points", type=int, default=60,
                    help="refuse to analyse fewer points than this (default "
                         "60). Below it the fit measures noise.")
    args = ap.parse_args()
    for path in args.logs:
        print("=" * 74)
        print(path)
        steps, losses = read_series(path)
        if Path(path).suffix != ".jsonl":
            alt = Path(path).with_suffix("").with_suffix(".jsonl")
            alt = Path(str(Path(path).with_suffix("")) + "_steps.jsonl")
            if alt.exists():
                s2, l2 = read_series(str(alt))
                if len(s2) > len(steps):
                    print(f"  (using the denser {alt.name})")
                    steps, losses = s2, l2
        if len(steps) < args.min_points:
            print(f"  only {len(steps)} points, below the {args.min_points} "
                  f"this analysis needs -- a fit to a handful of noisy "
                  f"points measures the noise, not the run (arm6 gave "
                  f"R2=0.03 from 20 --progress lines, which reads as 'no "
                  f"signal' rather than 'no progress'). Re-run with the "
                  f"harness, which writes a steps JSONL with EVERY step.")
            continue
        analyse(steps, losses, args.segments)
    return 0


if __name__ == "__main__":
    sys.exit(main())