"""The Batcher: where latents and prompts meet, and who owns the epoch.

``iter(batcher)`` is one epoch. It announces the epoch, brings the prompts up to date for it, runs a fresh
prefetch worker over the epoch's plan, and announces the end only after the last batch has been handed out,
so a handler of ``EpochEnding`` runs while nothing of the next epoch has been read.
"""

from __future__ import annotations

import random
from typing import Callable, Iterator

import torch

from ..graph.events import EpochEnding, EpochStarted, EventType
from .bucketing import Plan, bucket_map, make_plan, padding_stats
from .latent_cache import LatentSet
from .noise import Noiser
from .padding import FILLS, mirror_fill
from .prefetch import Prefetch
from .prompts import PromptTable
from .ids import image_of


class Batcher:
    def __init__(self, latents: LatentSet, prompts: PromptTable, noiser: Noiser, *, batch_size: int = 1,
                 shuffle: bool = True, keep_incomplete: bool = False, multiple: int = 1, max_shapes: int = 0, batch_area: float = 0.0, pack: float = 0.0, fill: str = "zero", seed: int = 0,
                 depth: int = 2, pin: bool = False, start_epoch: int = 0,
                 emit: Callable[[EventType], None] | None = None):
        orphans = [s for s in latents.samples if not prompts.knows(image_of(s))]
        if orphans:
            raise ValueError(f"{len(orphans)} sample(s) have latents but no prompt entry (first: {orphans[0]}); "
                             f"the prompt table must be built from the same images")
        self.latents, self.prompts, self.noiser = latents, prompts, noiser
        self.batch_size, self.shuffle, self.keep_incomplete = batch_size, shuffle, keep_incomplete
        self.multiple, self.max_shapes, self.seed, self.depth, self.pin = multiple, max_shapes, seed, depth, pin
        if fill not in FILLS:
            raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")
        self.batch_area, self.pack, self.fill = batch_area, pack, fill
        self.epoch = start_epoch
        self.emit = emit or (lambda event: None)
        self._factor = int(latents.spec["spatial_factor"])
        self.full_batch_area = int(batch_area * (latents.policy.base // self._factor) ** 2)
        self.plan_for(start_epoch)  # an impossible configuration fails now, not on the first step

    def plan_for(self, epoch: int) -> Plan:
        return make_plan(self.latents.shapes, batch_size=self.batch_size, multiple=self.multiple, max_shapes=self.max_shapes, full_batch_area=self.full_batch_area, pack=self.pack,
                         shuffle=self.shuffle, keep_incomplete=self.keep_incomplete,
                         rng=random.Random(f"{self.seed}:plan:{epoch}"))

    def shape_summary(self) -> dict:
        """What the model will see: distinct input shapes (batch size included, as a short batch is a shape of its own)
        and how much of the bucketed area is padding."""
        plan = self.plan_for(self.epoch)
        stats = padding_stats(list(self.latents.shapes.values()), self.multiple, self.max_shapes)
        pads = [1.0 - (h * w) / (shape[0] * shape[1]) for ids, shape in plan.batches for h, w in (self.latents.shapes[i] for i in ids)]
        return {"samples": stats["samples"], "shapes_in": stats["shapes_in"], "bucket_shapes": stats["shapes_out"],
                "batch_shapes": len({(len(ids), *shape) for ids, shape in plan.batches}), "batches": len(plan),
                "short_batches": sum(1 for ids, shape in plan.batches if len(ids) < self._full_size(shape)),
                "packed": plan.packed, "dropped": plan.dropped,
                "padding_mean": round(sum(pads) / max(len(pads), 1), 3), "padding_worst": round(max(pads, default=0.0), 3)}

    def _full_size(self, shape: tuple[int, int]) -> int:
        if not self.full_batch_area:
            return self.batch_size
        return max(1, min(self.batch_size, self.batch_size * self.full_batch_area // (shape[0] * shape[1])))

    def largest_units(self) -> int:
        """Latent pixels of the largest batch of the run (samples times the padded area)."""
        return max(len(ids) * shape[0] * shape[1] for ids, shape in self.plan_for(self.epoch).batches)

    def clean_item(self, sample: str) -> tuple[torch.Tensor, torch.Tensor | None, tuple[int, int]]:
        """A sample as the model sees it when nothing is random: its latent centred in its bucket shape, the validity
        mask (None without padding) and its true size in pixels."""
        latent = self.latents.get(sample)
        h, w = latent.shape[-2:]
        height, width = bucket_map(self.latents.shapes.values(), self.multiple, self.max_shapes)[(h, w)]
        top, left = (height - h) // 2, (width - w) // 2
        x0 = torch.zeros(1, latent.shape[0], height, width)
        x0[0, :, top:top + h, left:left + w] = latent
        mask = None
        if (height, width) != (h, w):
            mask = torch.zeros(1, 1, height, width)
            mask[..., top:top + h, left:left + w] = 1.0
        return x0, mask, (h * self._factor, w * self._factor)

    def max_batch_size(self) -> int:
        return max(len(ids) for ids, _ in self.plan_for(self.epoch).batches)

    def probe_batch(self, like: dict, count: int = 1) -> dict:
        """``count`` samples at the largest shape of the data, noised like a real batch, wearing the prompt of ``like``'s
        first sample: what a memory probe runs through the model."""
        ids, (height, width) = max(self.plan_for(self.epoch).batches, key=lambda b: b[1][0] * b[1][1])
        x0 = torch.zeros(count, int(self.latents.spec["channels"]), height, width)
        batch = self.noiser(x0, torch.Generator().manual_seed(self.seed))
        batch.update({"samples": [ids[i % len(ids)] for i in range(count)], "prompts": list(like["prompts"][:1]) * count, "epoch": self.epoch,
                      "prompts_version": like.get("prompts_version"), "sizes_px": [(height * self._factor, width * self._factor)] * count})
        return batch

    def __len__(self) -> int:
        """Batches per epoch (the same for every epoch)."""
        return len(self.plan_for(self.epoch))

    def __iter__(self) -> Iterator[dict]:
        epoch = self.epoch
        self.emit(EpochStarted(epoch))
        self.prompts.set_epoch(epoch)
        plan = self.plan_for(epoch)
        version, texts = self.prompts.snapshot()
        last = len(plan.batches) - 1
        worker = Prefetch(lambda: (self._make(i, ids, shape, epoch, version, texts, i == last)
                                   for i, (ids, shape) in enumerate(plan.batches)), self.depth)
        served = 0
        try:
            for batch in worker:
                served += len(batch["samples"])
                yield batch
        finally:
            worker.close()
        self.emit(EpochEnding(epoch, served))
        self.epoch = epoch + 1

    def _make(self, index: int, ids: tuple[str, ...], shape: tuple[int, int], epoch: int, version: int,
              texts: dict[str, str], last: bool = False) -> dict:
        rng = random.Random(f"{self.seed}:batch:{epoch}:{index}")
        generator = torch.Generator().manual_seed(rng.getrandbits(62))
        height, width = shape
        x0 = torch.zeros(len(ids), int(self.latents.spec["channels"]), height, width)
        valid = torch.zeros(len(ids), 1, height, width)
        true_sizes, placed = [], []
        for k, sample in enumerate(ids):
            latent = self.latents.get(sample)
            h, w = latent.shape[-2:]
            top = rng.randint(0, height - h) if height > h else 0
            left = rng.randint(0, width - w) if width > w else 0
            x0[k, :, top:top + h, left:left + w] = latent
            true_sizes.append((h * self._factor, w * self._factor))
            placed.append((top, left, h, w))
            valid[k, :, top:top + h, left:left + w] = 1.0
        batch = self.noiser(x0, generator)
        if self.fill == "mirror":
            for k, (top, left, h, w) in enumerate(placed):
                if (h, w) != (height, width):
                    batch["x_t"][k] = mirror_fill(batch["x_t"][k], top, left, h, w)
        batch.update({
            "samples": list(ids),
            "prompts": [texts[image_of(s)] for s in ids],
            "epoch": epoch,
            "last_in_epoch": last,
            "prompts_version": version,
            "sizes_px": true_sizes,
        })
        if self.pack > 0 or bool((valid == 0).any()):  # with packing every batch carries one, so shapes do not split by mask presence
            batch["valid_mask"] = valid
        if self.pin:
            batch = {k: v.pin_memory() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch
