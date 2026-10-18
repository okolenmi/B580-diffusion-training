"""Which samples share a batch.

Samples of the same latent shape batch together. With a bucket multiple above 1, shapes are rounded up
to the multiple and smaller latents are padded (the batcher does the padding and marks it in a valid
mask), so more samples share a shape and the model sees fewer distinct sizes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

CLUMP = 4  # same-shape batches kept together so the compiled kernels see the same size a few steps running


def bucket_shape(h: int, w: int, multiple: int) -> tuple[int, int]:
    if multiple <= 1:
        return h, w
    return -(-h // multiple) * multiple, -(-w // multiple) * multiple


def bucket_map(shapes: Iterable[tuple[int, int]], multiple: int, max_shapes: int = 0) -> dict[tuple[int, int], tuple[int, int]]:
    """Latent shape of a sample -> the shape it is padded to.

    Shapes are rounded up to ``multiple``; when more than ``max_shapes`` remain (0 means no limit) the two whose
    merge into their common bounding box adds the least padded area are merged, until the limit holds. Every new shape
    costs kernel compilation and driver memory the first time the model sees it, so a short list pays for itself.
    """
    counts: dict[tuple[int, int], int] = {}
    members: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for shape in shapes:
        target = bucket_shape(*shape, multiple)
        counts[target] = counts.get(target, 0) + 1
        members.setdefault(target, [])
        if shape not in members[target]:
            members[target].append(shape)
    groups = [(np.array(t), n, m) for t, n, m in ((t, counts[t], members[t]) for t in counts)]
    if max_shapes and len(groups) > max_shapes:
        hw = np.array([g[0] for g in groups], dtype=np.int64)
        n = np.array([g[1] for g in groups], dtype=np.int64)
        origin = [list(g[2]) for g in groups]
        while len(hw) > max_shapes:
            joined = np.maximum(hw[:, None, :], hw[None, :, :])
            area = joined[..., 0] * joined[..., 1]
            own = hw[:, 0] * hw[:, 1]
            cost = n[:, None] * (area - own[:, None]) + n[None, :] * (area - own[None, :])
            np.fill_diagonal(cost, np.iinfo(np.int64).max)
            i, j = divmod(int(cost.argmin()), len(hw))
            i, j = min(i, j), max(i, j)
            hw[i], n[i] = joined[i, j], n[i] + n[j]
            origin[i] += origin[j]
            hw, n = np.delete(hw, j, axis=0), np.delete(n, j)
            del origin[j]
        groups = [(hw[k], int(n[k]), origin[k]) for k in range(len(hw))]
    return {shape: (int(t[0]), int(t[1])) for t, _, origin in groups for shape in origin}


@dataclass(frozen=True)
class Plan:
    batches: tuple[tuple[tuple[str, ...], tuple[int, int]], ...]  # (sample ids, bucket latent shape)
    dropped: int  # samples left out because their group could not fill a batch

    def __len__(self) -> int:
        return len(self.batches)


def make_plan(shapes: Mapping[str, tuple[int, int]], *, batch_size: int, multiple: int = 1, max_shapes: int = 0, full_batch_area: int = 0,
              shuffle: bool, keep_incomplete: bool, rng: random.Random) -> Plan:
    """Group by bucket shape, cut into batches, and (when shuffling) order them in same-shape clumps.

    ``full_batch_area`` (0 means no limit) is the area one sample may take in a full batch, so a batch may hold
    ``batch_size * full_batch_area`` pixels: the batch shrinks to as many samples as fit (rounded down, at least one).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    target = bucket_map(shapes.values(), multiple, max_shapes)
    groups: dict[tuple[int, int], list[str]] = {}
    for sample in sorted(shapes):
        groups.setdefault(target[shapes[sample]], []).append(sample)
    batches: list[tuple[tuple[str, ...], tuple[int, int]]] = []
    dropped = 0
    for shape, members in groups.items():
        if shuffle:
            rng.shuffle(members)
        size = max(1, min(batch_size, batch_size * full_batch_area // (shape[0] * shape[1]))) if full_batch_area else batch_size
        for start in range(0, len(members), size):
            chunk = members[start:start + size]
            if len(chunk) < size and shuffle and not keep_incomplete:
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


def padding_stats(shapes: Sequence[tuple[int, int]], multiple: int, max_shapes: int = 0) -> dict:
    """How much of the bucketed area is padding: what a bucket multiple and shape limit cost on this dataset."""
    if not shapes:
        return {"samples": 0}
    target = bucket_map(shapes, multiple, max_shapes)
    fractions = sorted(1.0 - (h * w) / float(target[(h, w)][0] * target[(h, w)][1]) for h, w in shapes)
    n = len(fractions)
    per_shape: dict[tuple[int, int], int] = {}
    for shape in shapes:
        per_shape[shape] = per_shape.get(shape, 0) + 1
    return {
        "multiple": multiple, "samples": n,
        "shapes_in": len(per_shape), "shapes_out": len(set(target.values())),
        "padded_samples": sum(1 for f in fractions if f > 0),
        "pad_median": fractions[n // 2], "pad_mean": sum(fractions) / n,
        "pad_p90": fractions[min(n - 1, int(0.9 * n))], "pad_max": fractions[-1],
        "per_shape": [{"latent": list(k), "bucket": list(target[k]), "samples": c} for k, c in sorted(per_shape.items())],
    }


@dataclass(frozen=True)
class BucketPolicy:
    """The bucketing choices a Bucketer node hands to the batcher."""

    multiple: int = 1
    keep_incomplete: bool = False
    max_shapes: int = 0
    batch_area: float = 0.0  # multiples of the base resolution's area a sample may take and still run in a full batch; 0 = no limit
