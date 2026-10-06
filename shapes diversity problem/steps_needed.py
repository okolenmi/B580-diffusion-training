"""How many steps does a real training run need before a revisit can show up?

The 40-step hw_validate comparison on `non-square` came out identical with and
without a bigger primitive cache (0.413 vs 0.411 steps/sec), which reads as
"the fix does not work on the training path". It cannot conclude that: the
dataset now has 63 distinct latent shapes, so in 40 steps almost every step is
a *first sighting*, and a first sighting is slow for reasons a bigger cache
does not remove. The discriminator needs revisits, and there are barely any.

This asks the cheap question first, with no GPU at all: given the real batch
order this run produces, how many revisits exist by step N? That sets the step
count for the next run instead of guessing.

The batch order is taken from the loader itself, seeded the way hw_validate
seeds it, so the shape sequence here is the one the trainer sees.
"""

import sys
from pathlib import Path

REPO = Path("/home/okolenmi/Desktop/B580-diffusion-training")
sys.path.insert(0, str(REPO))


def shape_of(batch):
    """The latent shape of one batch, however this loader spells it."""
    for key in ("latent_size", "size", "shape", "resolution"):
        v = batch.get(key) if isinstance(batch, dict) else None
        if v is None:
            continue
        if isinstance(v, (tuple, list)) and len(v) == 2:
            return f"{int(v[0])}x{int(v[1])}"
        if isinstance(v, str) and "x" in v:
            return v
    lat = batch.get("latents") if isinstance(batch, dict) else None
    if lat is not None and hasattr(lat, "shape") and len(lat.shape) >= 3:
        return f"{lat.shape[-2]}x{lat.shape[-1]}"
    return None


def main() -> int:
    from nodes.dataset.managed import ManagedDatasetSourceNode

    node = ManagedDatasetSourceNode(None).build(
        dataset_root="non-square", batch_size=2, shuffle=True,
        keep_incomplete_batches=False,
    )["batches"]
    total = len(node)
    shapes = []
    for i, b in enumerate(node):
        s = shape_of(b)
        if s is None:
            print(f"could not read a latent shape from batch {i}; keys: "
                  f"{sorted(b)[:12] if isinstance(b, dict) else type(b)}")
            return 2
        shapes.append(s)

    print(f"epoch: {total} batches, {len(set(shapes))} distinct shapes")
    print()
    print("cumulative shape coverage and revisits:")
    print(f"{'steps':>7}  {'distinct':>9}  {'revisits':>9}  {'% revisit':>10}")
    seen = set()
    revisits = 0
    marks = [10, 20, 40, 60, 80, 100, 136, 200, 273, 400, 546]
    for i, s in enumerate(shapes * 4):
        if i > 546:
            break
        if s in seen:
            revisits += 1
        else:
            seen.add(s)
        if (i + 1) in marks:
            print(f"{i + 1:>7}  {len(seen):>9}  {revisits:>9}  "
                  f"{revisits / (i + 1):>9.0%}")

    # The threshold that matters: the step at which half the steps are revisits.
    half = None
    seen.clear()
    for i, s in enumerate(shapes * 8):
        if i > 2000:
            break
        is_revisit = s in seen
        seen.add(s)
        if is_revisit and half is None and (i + 1 - len(seen)) / (i + 1) >= 0.5:
            half = i + 1
            break
    print()
    print(f"first step count at which revisits are half of all steps: {half}")
    print()
    print("first 40 step shapes:", shapes[:40])
    print("revisits within the first 40 steps:",
          sum(1 for s in shapes[:40] if shapes[:40].index(s) != shapes[:40].index(s)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
