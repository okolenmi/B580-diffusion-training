"""`_fit_crop_boxes` -- the crop arithmetic that bounds a long side.

Found by reading rather than by a failure: the docstring on this method
said its own checks were never shipped. It is pure arithmetic, needs no
image data and no device, and it is the part of ingestion most likely to
be edited wrongly -- a crop that leaves a gap loses pixels silently, and
one that overhangs the image crashes on the first batch.

The properties that matter, and why:

* **every box is inside the image.** An overhang is a crash, not a
  cosmetic error.
* **every side is a multiple of 8.** VAE downsampling requires it; this is
  the same rule "fit" mode already had.
* **no box exceeds the cap.** That is the entire point -- an oversized box
  is the VRAM ratchet this method exists to prevent.
* **the boxes tile the long side with no gap.** Overlap and gaps both
  change what the dataset is, silently.
* **equal segments, not n-1 plus a remainder.** A trailing crop a tenth
  the size of the others is a different amount of signal per sample, which
  is not what "split this image" should mean.

Run directly: python nodes/smoke_tests/smoke_test_fit_crop_boxes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from manager.builder import DataTaskRunner  # noqa: E402

CAP = 1024


def _boxes(w: int, h: int, px: int = CAP, cap_ratio: float = 1.0):
    """Call the method without constructing a runner.

    `_fit_crop_boxes` is pure arithmetic and reads nothing off `self`, so
    building a DataTaskRunner (which wants a dataset root, a config and a
    library) to reach it would be ceremony for nothing. `None` stands in
    for the runner, and the test fails loudly rather than silently if a
    future edit makes the method actually use `self`.
    """
    return DataTaskRunner._fit_crop_boxes(None, w, h, px, cap_ratio)


def invariants(label: str, w: int, h: int, px: int, cap_ratio: float) -> None:
    boxes, rw, rh = _boxes(w, h, px, cap_ratio)
    cap = round(cap_ratio * px)
    assert boxes, f"{label}: produced no boxes"

    for i, (l, t, r, b) in enumerate(boxes):
        assert 0 <= l < r <= rw and 0 <= t < b <= rh, (
            f"{label}: box {i} is outside the {rw}x{rh} image: {(l, t, r, b)}")
        assert (r - l) % 8 == 0 and (b - t) % 8 == 0, (
            f"{label}: box {i} is not /8-aligned: {r - l}x{b - t}")
        assert max(r - l, b - t) <= cap, (
            f"{label}: box {i} exceeds the {cap}px cap: {r - l}x{b - t}")

    # Tiling along the long axis. Not *exact* tiling, and cannot be: each
    # box is snapped down to a multiple of 8 independently, so every
    # boundary drops the remainder -- up to 7 px per seam. What must hold
    # is that the seams are small and never negative, because a negative
    # one is an overlap and an overlap means the same pixels appear in two
    # dataset samples. (The docstring on the method called this
    # "full-coverage"; it is coverage to within the /8 snap, and saying so
    # is the difference between a claim and a fact.)
    axis_is_x = rw >= rh
    ordered = sorted(boxes, key=lambda bx: bx[0] if axis_is_x else bx[1])
    for i in range(len(ordered) - 1):
        prev_end = ordered[i][2] if axis_is_x else ordered[i][3]
        next_start = ordered[i + 1][0] if axis_is_x else ordered[i + 1][1]
        seam = next_start - prev_end
        assert 0 <= seam <= 7, (
            f"{label}: seam {i} is {seam}px -- negative means the boxes "
            f"overlap, more than 7 means something other than /8 snapping "
            f"is going on: {ordered}")

    # Equal segments, not n-1 full crops plus a small remainder.
    if len(boxes) > 1:
        spans = [max(bb[2] - bb[0], bb[3] - bb[1]) for bb in boxes]
        assert min(spans) >= max(spans) - 8, (
            f"{label}: box sizes are not equal -- shortest {min(spans)}, "
            f"longest {max(spans)}: {boxes}")

    print(f"    PASS {label}: {len(boxes)} box(es) over {rw}x{rh}, cap {cap}, "
          f"long axis {rw if axis_is_x else rh}px")


def main() -> None:
    print("[inside the cap: one box, the 'fit' behaviour]")
    invariants("square 512", 512, 512, CAP, 1.0)
    invariants("tall 512x1024", 512, 1024, CAP, 1.0)
    invariants("wide 1024x512", 1024, 512, CAP, 1.0)

    print("[over the cap: split along the long side]")
    # A 500x1180 source is the real shape that motivated this method:
    # ~2.34x, so at px=1024 the long side lands at 2417 and must be tiled.
    invariants("500x1180 (the motivating case)", 500, 1180, CAP, 1.0)
    invariants("extreme 256x4096", 256, 4096, CAP, 1.0)
    invariants("odd sizes 777x1234", 777, 1234, CAP, 1.0)
    invariants("tiny 8x8", 8, 8, CAP, 1.0)

    print("[a fractional cap still lands on whole /8 boxes]")
    invariants("500x1180 at cap_ratio 1.5", 500, 1180, CAP, 1.5)
    invariants("256x4096 at cap_ratio 2.0", 256, 4096, CAP, 2.0)

    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()