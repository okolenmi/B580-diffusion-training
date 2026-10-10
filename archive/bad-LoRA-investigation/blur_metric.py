#!/usr/bin/env python3
"""Laplacian variance (sharpness) for B-render comparisons.

Usage: python blur_metric.py <image1> [<image2> ...]
Needs: Pillow, numpy. No torch, no GPU.

Reports per image: full-frame Laplacian variance + center-crop variance
(higher = sharper). Relative comparisons between arms of the same scene
are the point; absolute numbers mean little across different content.
"""
import sys

import numpy as np
from PIL import Image


def lap_var(gray: np.ndarray) -> float:
    lap = (np.roll(gray, 1, 0) + np.roll(gray, -1, 0)
           + np.roll(gray, 1, 1) + np.roll(gray, -1, 1)
           - 4.0 * gray)
    return float(lap[1:-1, 1:-1].var())


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python blur_metric.py <image> [...]")
        return 1
    print(f"{'image':>40} {'full':>10} {'center':>10}")
    for path in sys.argv[1:]:
        img = Image.open(path).convert("L")
        g = np.asarray(img, dtype=np.float64)
        h, w = g.shape
        c = g[h // 4:3 * h // 4, w // 4:3 * w // 4]
        print(f"{path[-40:]:>40} {lap_var(g):>10.1f} {lap_var(c):>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
