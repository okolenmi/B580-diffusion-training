#!/usr/bin/env python3
"""Choose a small set of canonical latent shapes for a multi-resolution dataset.

Why. On the Intel B580 every distinct latent shape that reaches the device is
a separate specialisation (the project measured 44 shapes -> 0.412 steps/s
against 0.813 for one shape). Cropping or padding every sample to one of a
few canonical shapes collapses that. This module decides *which* shapes and
*what each sample does* (crop, pad or both), from the dataset's own shape
histogram, under limits you set.

Policy, per dimension. A sample side `h` is assigned a canonical side `H`
(a multiple of `quantum`) with `h - max_crop <= H <= h + max_pad`:
  * H < h : the side is CROPPED by h - H latent px (random offset each epoch,
            so over many epochs every pixel is seen; SDXL crop conditioning
            is updated to match -- see `crop_conditioning`).
  * H > h : the side is PADDED by H - h latent px (needs a loss mask).
  * H == h: untouched.
`max_pad=0` (the default) means crop-only, which needs no loss mask and keeps
the loss exactly what it is today. Padding is opt-in.

The 1-D problem (choose k canonical values for one axis) is solved exactly by
dynamic programming over contiguous groups of sorted sizes (optimal groups in
one dimension are contiguous). The 2-D plan combines an H set and a W set and
reports the number of distinct (H, W) pairs the dataset actually produces.

Usage:
    python3 shape_policy.py --dataset datasets/non-square --batch-size 2
    python3 shape_policy.py --shapes "64x96:120,96x64:80,72x88:3" --max-crop 16
Library:
    plan(...), apply_policy(latent, H, W, rng), crop_conditioning(...)
"""
from __future__ import annotations

import argparse
import itertools
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

INF = float("inf")


# ---------------------------------------------------------------- 1-D solver

def _side_cost(h: int, H: int, crop_w: float, pad_w: float) -> float:
    """Cost of mapping side h onto canonical side H (relative, dimensionless)."""
    if H < h:
        return crop_w * (h - H) / h
    return pad_w * (H - h) / h


def _feasible(h: int, H: int, max_crop: int, max_pad: int) -> bool:
    return h - max_crop <= H <= h + max_pad


def best_1d(
    counts: Mapping[int, int], k: int, *, quantum: int, max_crop: int,
    max_pad: int, crop_w: float = 1.0, pad_w: float = 0.5,
) -> tuple[float, dict[int, int]] | None:
    """Exact best assignment of every size in `counts` to at most k canonical
    values (multiples of `quantum`). Returns (total weighted cost, mapping
    size -> canonical) or None if k values cannot cover everything within the
    crop/pad limits."""
    sizes = sorted(counts)
    n = len(sizes)
    lo = max(quantum, (min(sizes) - max_crop) // quantum * quantum)
    hi = (max(sizes) + max_pad + quantum - 1) // quantum * quantum
    candidates = list(range(lo, hi + 1, quantum))

    # group_cost[i][j]: best (cost, H) for sizes[i..j] sharing one canonical H
    group: list[list[tuple[float, int]]] = [[(INF, 0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            best = (INF, 0)
            for H in candidates:
                total = 0.0
                ok = True
                for s in sizes[i:j + 1]:
                    if not _feasible(s, H, max_crop, max_pad):
                        ok = False
                        break
                    total += counts[s] * _side_cost(s, H, crop_w, pad_w)
                if ok and total < best[0]:
                    best = (total, H)
            group[i][j] = best

    # dp[g][j]: min cost covering sizes[0..j] with g groups
    dp = [[INF] * n for _ in range(k + 1)]
    cut = [[-1] * n for _ in range(k + 1)]
    for j in range(n):
        dp[1][j] = group[0][j][0]
    for g in range(2, k + 1):
        for j in range(n):
            for i in range(1, j + 1):
                c = dp[g - 1][i - 1] + group[i][j][0]
                if c < dp[g][j]:
                    dp[g][j], cut[g][j] = c, i
    g_best = min(range(1, k + 1), key=lambda g: dp[g][n - 1])
    if dp[g_best][n - 1] == INF:
        return None
    mapping: dict[int, int] = {}
    j, g = n - 1, g_best
    while g >= 1:
        i = 0 if g == 1 else cut[g][j]
        H = group[i][j][1]
        for s in sizes[i:j + 1]:
            mapping[s] = H
        j, g = i - 1, g - 1
    return dp[g_best][n - 1], mapping


# ---------------------------------------------------------------- 2-D plan

@dataclass(frozen=True)
class Candidate:
    k_h: int
    k_w: int
    shapes: dict[tuple[int, int], int]          # canonical (H, W) -> samples
    mapping: dict[tuple[int, int], tuple[int, int]]  # (h, w) -> (H, W)
    compute_factor: float      # mean (H*W)/(h*w) over samples: <1 faster, >1 slower
    kept_fraction: float       # mean fraction of each sample's area still seen per epoch
    worst_crop_px: int
    worst_pad_px: int
    batches_per_shape: float   # mean full batches per canonical shape (clump size proxy)
    padded_samples: int

    @property
    def n_shapes(self) -> int:
        return len(self.shapes)


def plan(
    shape_counts: Mapping[tuple[int, int], int], *, quantum: int = 8,
    max_crop: int = 16, max_pad: int = 0, max_k: int = 6,
    crop_w: float = 1.0, pad_w: float = 0.5, batch_size: int = 1,
) -> list[Candidate]:
    """All (k_h, k_w) combinations up to max_k, as candidates sorted by number
    of shapes then cost. Infeasible combinations are dropped."""
    if not shape_counts:
        raise ValueError("no shapes")
    for (h, w), c in shape_counts.items():
        if h <= 0 or w <= 0 or c <= 0:
            raise ValueError(f"bad shape entry {(h, w)}: {c}")
    h_counts: dict[int, int] = {}
    w_counts: dict[int, int] = {}
    for (h, w), c in shape_counts.items():
        h_counts[h] = h_counts.get(h, 0) + c
        w_counts[w] = w_counts.get(w, 0) + c
    h_solutions = {k: best_1d(h_counts, k, quantum=quantum, max_crop=max_crop,
                              max_pad=max_pad, crop_w=crop_w, pad_w=pad_w)
                   for k in range(1, min(max_k, len(h_counts)) + 1)}
    w_solutions = {k: best_1d(w_counts, k, quantum=quantum, max_crop=max_crop,
                              max_pad=max_pad, crop_w=crop_w, pad_w=pad_w)
                   for k in range(1, min(max_k, len(w_counts)) + 1)}
    total = sum(shape_counts.values())
    out: list[Candidate] = []
    seen: set[frozenset] = set()
    for kh, kw in itertools.product(h_solutions, w_solutions):
        hs, ws = h_solutions[kh], w_solutions[kw]
        if hs is None or ws is None:
            continue
        hmap, wmap = hs[1], ws[1]
        mapping = {(h, w): (hmap[h], wmap[w]) for (h, w) in shape_counts}
        key = frozenset(mapping.items())
        if key in seen:
            continue
        seen.add(key)
        shapes: dict[tuple[int, int], int] = {}
        comp = kept = 0.0
        crop_px = pad_px = padded = 0
        for (h, w), c in shape_counts.items():
            H, W = mapping[(h, w)]
            shapes[(H, W)] = shapes.get((H, W), 0) + c
            comp += c * (H * W) / (h * w)
            kept += c * (min(h, H) / h) * (min(w, W) / w)
            crop_px = max(crop_px, max(0, h - H), max(0, w - W))
            pad_px = max(pad_px, max(0, H - h), max(0, W - w))
            if H > h or W > w:
                padded += c
        full_batches = [c // batch_size for c in shapes.values()]
        out.append(Candidate(
            k_h=kh, k_w=kw, shapes=shapes, mapping=mapping,
            compute_factor=comp / total, kept_fraction=kept / total,
            worst_crop_px=crop_px, worst_pad_px=pad_px,
            batches_per_shape=sum(full_batches) / len(shapes),
            padded_samples=padded))
    out.sort(key=lambda c: (c.n_shapes, -c.kept_fraction, c.compute_factor))
    return out


def baseline(shape_counts: Mapping[tuple[int, int], int], batch_size: int = 1) -> Candidate:
    """The as-is policy, for the comparison row."""
    shapes = dict(shape_counts)
    return Candidate(
        k_h=0, k_w=0, shapes=shapes, mapping={s: s for s in shape_counts},
        compute_factor=1.0, kept_fraction=1.0, worst_crop_px=0, worst_pad_px=0,
        batches_per_shape=sum(c // batch_size for c in shapes.values()) / len(shapes),
        padded_samples=0)


def pad_up(shape_counts: Mapping[tuple[int, int], int], multiple: int, batch_size: int = 1) -> Candidate:
    """The simple 'round each side up to a multiple' policy the project measured."""
    up = lambda v: -(-v // multiple) * multiple
    mapping = {(h, w): (up(h), up(w)) for (h, w) in shape_counts}
    shapes: dict[tuple[int, int], int] = {}
    total = sum(shape_counts.values())
    comp = pad_px = padded = 0
    for (h, w), c in shape_counts.items():
        H, W = mapping[(h, w)]
        shapes[(H, W)] = shapes.get((H, W), 0) + c
        comp += c * H * W / (h * w)
        pad_px = max(pad_px, H - h, W - w)
        padded += c if (H, W) != (h, w) else 0
    return Candidate(
        k_h=0, k_w=0, shapes=shapes, mapping=mapping, compute_factor=comp / total,
        kept_fraction=1.0, worst_crop_px=0, worst_pad_px=pad_px,
        batches_per_shape=sum(c // batch_size for c in shapes.values()) / len(shapes),
        padded_samples=padded)


def recommend(cands: Sequence[Candidate], *, max_shapes: int = 8, min_kept: float = 0.90) -> Candidate | None:
    """The candidate with the fewest shapes that keeps >= min_kept of the
    content per epoch, preferring the cheaper one on ties. None if none fits."""
    ok = [c for c in cands if c.n_shapes <= max_shapes and c.kept_fraction >= min_kept]
    if not ok:
        return None
    fewest = min(c.n_shapes for c in ok)
    return min((c for c in ok if c.n_shapes == fewest),
               key=lambda c: (c.compute_factor, -c.kept_fraction))


# ------------------------------------------------- applying a policy (training)

def apply_policy(latent, H: int, W: int, rng, *, pad_value: float = 0.0):
    """Crop and/or pad one (C, h, w) latent tensor to (C, H, W).

    Returns (out, info) with info = dict(crop_top, crop_left, pad_top, pad_left,
    valid) where `valid` is a (H, W) bool tensor, True where the pixel is real
    (all True for crop-only). Crop offsets are drawn from `rng` (a
    `random.Random`) so a seeded epoch is reproducible. Padding is placed
    uniformly at random as well, so the model sees borders on every side.
    """
    import torch  # local: the planner itself needs no torch

    C, h, w = latent.shape
    ch, cw = max(0, h - H), max(0, w - W)       # pixels to crop away
    ph, pw = max(0, H - h), max(0, W - w)       # pixels to pad on
    top = rng.randint(0, ch) if ch else 0
    left = rng.randint(0, cw) if cw else 0
    kept = latent[:, top:top + min(h, H), left:left + min(w, W)]
    out = torch.full((C, H, W), pad_value, dtype=latent.dtype, device=latent.device)
    valid = torch.zeros((H, W), dtype=torch.bool, device=latent.device)
    pt = rng.randint(0, ph) if ph else 0
    pl = rng.randint(0, pw) if pw else 0
    out[:, pt:pt + kept.shape[1], pl:pl + kept.shape[2]] = kept
    valid[pt:pt + kept.shape[1], pl:pl + kept.shape[2]] = True
    return out, dict(crop_top=top, crop_left=left, pad_top=pt, pad_left=pl, valid=valid)


def crop_conditioning(h: int, w: int, info: Mapping[str, int], vae_scale: int = 8) -> dict[str, int]:
    """Arguments for `resolution_embedding` (SDXL micro-conditioning) for a
    cropped sample: original size is the uncropped one, the crop offsets are
    the ones drawn, and the target size is the size actually fed to the model.
    All in image pixels (latent px * vae_scale)."""
    return dict(
        height=h * vae_scale, width=w * vae_scale,
        crop_h=info["crop_top"] * vae_scale, crop_w=info["crop_left"] * vae_scale,
    )


# ---------------------------------------------------------------- input / CLI

def read_dataset_shapes(dataset_dir: str | Path) -> dict[tuple[int, int], int]:
    """(latent_h, latent_w) -> sample count from a v2 dataset's metadata.db.
    Opened read-only; legacy v1 datasets are not supported here."""
    db = Path(dataset_dir) / "metadata.db"
    if not db.is_file():
        raise FileNotFoundError(f"{db} not found")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT latent_h, latent_w, COUNT(*) FROM trajectories "
            "WHERE latent_h > 0 AND latent_w > 0 GROUP BY latent_h, latent_w").fetchall()
    finally:
        con.close()
    return {(int(h), int(w)): int(c) for h, w, c in rows}


def parse_shapes(text: str) -> dict[tuple[int, int], int]:
    out: dict[tuple[int, int], int] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        dims, _, count = part.partition(":")
        h, _, w = dims.lower().partition("x")
        out[(int(h), int(w))] = int(count or 1)
    return out


def _row(label: str, c: Candidate) -> str:
    return (f"{label:<26}{c.n_shapes:>7}{c.compute_factor:>10.2f}x{c.kept_fraction:>9.1%}"
            f"{c.worst_crop_px:>8}{c.worst_pad_px:>7}{c.batches_per_shape:>11.1f}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="dataset directory containing metadata.db (v2)")
    src.add_argument("--shapes", help='"HxW:count,HxW:count" (latent px)')
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--quantum", type=int, default=8, help="canonical sides are multiples of this (latent px)")
    ap.add_argument("--max-crop", type=int, default=16, help="most latent px cropped from one side")
    ap.add_argument("--max-pad", type=int, default=0, help="most latent px padded on one side (0 = crop-only)")
    ap.add_argument("--max-k", type=int, default=6, help="most canonical values per axis to try")
    ap.add_argument("--max-shapes", type=int, default=8)
    ap.add_argument("--min-kept", type=float, default=0.90, help="least mean content fraction kept per epoch")
    ap.add_argument("--all", action="store_true", help="print every candidate, not just the best per shape count")
    args = ap.parse_args(argv)

    counts = read_dataset_shapes(args.dataset) if args.dataset else parse_shapes(args.shapes)
    total = sum(counts.values())
    print(f"{total} samples, {len(counts)} distinct latent shapes, "
          f"H {min(h for h, _ in counts)}-{max(h for h, _ in counts)}, "
          f"W {min(w for _, w in counts)}-{max(w for _, w in counts)}  (batch size {args.batch_size})\n")
    head = f"{'policy':<26}{'shapes':>7}{'compute':>11}{'kept':>9}{'crop':>8}{'pad':>7}{'batch/shp':>11}"
    print(head + "\n" + "-" * len(head))
    print(_row("as-is", baseline(counts, args.batch_size)))
    for m in (32, 64):
        print(_row(f"pad up to x{m} (padded)", pad_up(counts, m, args.batch_size)))
    cands = plan(counts, quantum=args.quantum, max_crop=args.max_crop, max_pad=args.max_pad,
                 max_k=args.max_k, batch_size=args.batch_size)
    best_by_n: dict[int, Candidate] = {}
    for c in cands:
        cur = best_by_n.get(c.n_shapes)
        if cur is None or (c.kept_fraction, -c.compute_factor) > (cur.kept_fraction, -cur.compute_factor):
            best_by_n[c.n_shapes] = c
    shown = cands if args.all else [best_by_n[n] for n in sorted(best_by_n)][:12]
    kind = "crop" if args.max_pad == 0 else f"crop<={args.max_crop}/pad<={args.max_pad}"
    for c in shown:
        print(_row(f"{kind} q{args.quantum} k={c.k_h}x{c.k_w}", c))
    rec = recommend(cands, max_shapes=args.max_shapes, min_kept=args.min_kept)
    print()
    if rec is None:
        print(f"No candidate keeps >= {args.min_kept:.0%} of the content within {args.max_shapes} shapes; "
              f"raise --max-crop, allow --max-pad, or lower --min-kept.")
        return 1
    print(f"Recommended: {rec.n_shapes} shape(s), compute x{rec.compute_factor:.2f}, "
          f"kept {rec.kept_fraction:.1%}, worst crop {rec.worst_crop_px} px, worst pad {rec.worst_pad_px} px")
    for (H, W), n in sorted(rec.shapes.items(), key=lambda kv: -kv[1]):
        print(f"   {H:>3}x{W:<3} latent  ({H * 8}x{W * 8} px)  {n:>5} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
