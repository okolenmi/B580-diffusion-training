"""Which samples share a batch.

Samples of the same latent shape batch together. With a bucket multiple above 1, shapes are rounded up
to the multiple and smaller latents are padded (the batcher does the padding and marks it in a valid
mask), so more samples share a shape and the model sees fewer distinct sizes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

CLUMP = 4  # same-shape batches kept together so the compiled kernels see the same size a few steps running


def bucket_shape(h: int, w: int, multiple: int) -> tuple[int, int]:
    if multiple <= 1:
        return h, w
    return -(-h // multiple) * multiple, -(-w // multiple) * multiple


@dataclass(frozen=True)
class Plan:
    batches: tuple[tuple[tuple[str, ...], tuple[int, int]], ...]  # (sample ids, bucket latent shape)
    dropped: int  # samples left out because their group could not fill a batch

    def __len__(self) -> int:
        return len(self.batches)


def make_plan(shapes: Mapping[str, tuple[int, int]], *, batch_size: int, multiple: int = 1, shuffle: bool,
              keep_incomplete: bool, rng: random.Random) -> Plan:
    """Group by bucket shape, cut into batches, and (when shuffling) order them in same-shape clumps."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    groups: dict[tuple[int, int], list[str]] = {}
    for sample in sorted(shapes):
        groups.setdefault(bucket_shape(*shapes[sample], multiple), []).append(sample)
    batches: list[tuple[tuple[str, ...], tuple[int, int]]] = []
    dropped = 0
    for shape, members in groups.items():
        if shuffle:
            rng.shuffle(members)
        for start in range(0, len(members), batch_size):
            chunk = members[start:start + batch_size]
            if len(chunk) < batch_size and shuffle and not keep_incomplete:
                dropped += len(chunk)
                continue
            batches.append((tuple(chunk), shape))
    if not batches:
        raise ValueError(
            f"{len(shapes)} sample(s) form no batch: batch_size={batch_size} with shuffle drops every group "
            f"smaller than a batch. Lower batch_size, add images, keep incomplete batches, or turn shuffle off.")
    if not shuffle:
        return Plan(tuple(batches), dropped)
    by_shape: dict[tuple[int, int], list] = {}
    for batch in batches:
        by_shape.setdefault(batch[1], []).append(batch)
    clumps = []
    for group in by_shape.values():
        rng.shuffle(group)
        clumps += [group[i:i + CLUMP] for i in range(0, len(group), CLUMP)]
    rng.shuffle(clumps)
    return Plan(tuple(b for clump in clumps for b in clump), dropped)


def padding_stats(shapes: Sequence[tuple[int, int]], multiple: int) -> dict:
    """How much of the bucketed area is padding: what a bucket multiple costs on this dataset."""
    if not shapes:
        return {"samples": 0}
    fractions = sorted(1.0 - (h * w) / float(bh * bw) for h, w in shapes
                       for bh, bw in [bucket_shape(h, w, multiple)])
    n = len(fractions)
    per_shape: dict[tuple[int, int], list] = {}
    for h, w in shapes:
        per_shape.setdefault((h, w), []).append(bucket_shape(h, w, multiple))
    return {
        "multiple": multiple, "samples": n,
        "shapes_in": len(per_shape), "shapes_out": len({b[0] for b in per_shape.values()}),
        "padded_samples": sum(1 for f in fractions if f > 0),
        "pad_median": fractions[n // 2], "pad_mean": sum(fractions) / n,
        "pad_p90": fractions[min(n - 1, int(0.9 * n))], "pad_max": fractions[-1],
        "per_shape": [{"latent": list(k), "bucket": list(v[0]), "samples": len(v)} for k, v in sorted(per_shape.items())],
    }


@dataclass(frozen=True)
class BucketPolicy:
    """The bucketing choices a Bucketer node hands to the batcher."""

    multiple: int = 1
    keep_incomplete: bool = False
