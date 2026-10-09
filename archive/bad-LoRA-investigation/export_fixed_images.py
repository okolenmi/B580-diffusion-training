#!/usr/bin/env python3
"""Export the source images at a fixed PX x PX, for Kohya's arm.

Same arithmetic as export_pad1024_images.py (manager/builder.py's "pad"
branch: scale by px/max(w,h), centre-pad to px x px), parameterised on PX
so one fixed canvas can be produced at whichever resolution Kohya can
actually fit on this card.

PX is an argument rather than a constant because 1024 was measured, not
assumed: this trainer completes at 1024x1024 (10,635 MB peak) while kohya
dies at step 0 there (10,268 MB before teardown) even with bucketing off
and identical geometry. 768 is the next size down to try.

Verified on write: every output must be exactly PX x PX. A silent
off-by-one would reintroduce exactly the geometry mismatch that made the
first kohya attempt train on up-to-1408px canvases.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
SRC = Path("/home/okolenmi/Downloads/datas")


def main() -> int:
    px = int(sys.argv[1]) if len(sys.argv) > 1 else 768
    out = REPO / "runs" / f"a2_kohya_{px}img"
    from PIL import Image

    if not out.exists():
        out.mkdir(parents=True)
    n = 0
    for p in sorted(SRC.iterdir()):
        if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        with Image.open(p) as im:
            img = im.convert("RGB")
            w, h = img.size
            scale = px / max(w, h)
            nw, nh = round(w * scale), round(h * scale)
            resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (px, px), (0, 0, 0))
            canvas.paste(resized, ((px - nw) // 2, (px - nh) // 2))
            canvas.save(out / f"{p.stem}.png")
            n += 1
    print(f"wrote {n} images at {px}x{px} -> {out}")

    sizes = set()
    for p in out.glob("*.png"):
        with Image.open(p) as im:
            sizes.add(im.size)
    print(f"distinct sizes: {sorted(sizes)}")
    if sizes != {(px, px)}:
        print(f"FAIL: expected only {(px, px)}")
        return 1

    meta = REPO / "runs" / f"a2_kohya_{px}meta.jsonl"
    with meta.open("w") as fh:
        for p in sorted(out.glob("*.png")):
            fh.write(f'{{"image_path": "{p}", "caption": "style"}}\n')
    print(f"metadata -> {meta} ({sum(1 for _ in meta.open())} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())