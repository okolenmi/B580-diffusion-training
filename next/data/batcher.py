"""The Batcher: where latents and prompts meet, and who owns the epoch.

``iter(batcher)`` is one epoch. It announces the epoch, brings the prompts up to date for it, runs a fresh
prefetch worker over the epoch's plan, and announces the end only after the last batch has been handed out,
so a handler of ``EpochEnding`` runs while nothing of the next epoch has been read.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator

import torch

from ..graph.events import EpochEnding, EpochStarted, EventType
from .bucketing import Plan, bucket_map, make_plan, padding_stats
from .latent_cache import LatentSet
from .noise import Noiser
from .padding import FILLS, mirror_fill
from .prefetch import Prefetch
from .ramp import BatchRamp
from .prompts import PromptTable
from .ids import image_of


class Batcher:
    def __init__(self, latents: LatentSet, prompts: PromptTable, noiser: Noiser, *, batch_size: int = 1,
                 shuffle: bool = True, keep_incomplete: bool = False, multiple: int = 1, max_shapes: int = 0, batch_area: float = 0.0, pack: float = 0.0, fill: str = "zero", seed: int = 0,
                 depth: int = 2, pin: bool = False, start_epoch: int = 0, skip: int = 0, ramp: BatchRamp | None = None,
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
        if skip and start_epoch:
            raise ValueError("start at an epoch or after a number of batches, not both")
        self.epoch, self.ramp = start_epoch, ramp
        self._plans: dict[int, Plan] = {}
        self._before: dict[int, int] = {0: 0}
        self.position, self._offset = skip, 0  # batches handed out since the start of the run; those of a first, partly done epoch to pass over
        self.emit = emit or (lambda event: None)
        self._factor = int(latents.spec["spatial_factor"])
        self.full_batch_area = int(batch_area * (latents.policy.base // self._factor) ** 2)
        self.plan_for(start_epoch)  # an impossible configuration fails now, not on the first step
        if skip:
            self.epoch = 0
            while self.batches_before(self.epoch + 1) <= skip:
                self.epoch += 1
            self._offset = skip - self.batches_before(self.epoch)

    def base_plan(self, epoch: int) -> Plan:
        """The epoch's batches at the full batch size (what the memory planning sizes for)."""
        if epoch not in self._plans:
            self._plans[epoch] = make_plan(self.latents.shapes, batch_size=self.batch_size, multiple=self.multiple, max_shapes=self.max_shapes,
                                           full_batch_area=self.full_batch_area, pack=self.pack, shuffle=self.shuffle, keep_incomplete=self.keep_incomplete,
                                           rng=random.Random(f"{self.seed}:plan:{epoch}"))
        return self._plans[epoch]

    def plan_for(self, epoch: int) -> Plan:
        """The epoch's batches; with a ramp the first ones of the run are cut smaller."""
        plan = self.base_plan(epoch)
        return plan if self.ramp is None else plan.split_front(self.ramp.size, self.ramp.left(self.batches_before(epoch)))

    def batches_before(self, epoch: int) -> int:
        """Batches handed out in the epochs before ``epoch``."""
        done = max(e for e in self._before if e <= epoch)
        for e in range(done, epoch):
            self._before[e + 1] = self._before[e] + len(self.plan_for(e))
        return self._before[epoch]

    def batches_in(self, epochs: int) -> int:
        """Batches of ``epochs`` epochs from the current one."""
        return self.batches_before(self.epoch + epochs) - self.batches_before(self.epoch)

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

    def largest_units(self, regions: int | None = None) -> int:
        """Latent pixels of the largest batch of the run (samples times the padded area). ``regions`` replaces the copies per
        image: 1 for a run that puts each region through the network on its own."""
        return (regions or self.noiser.max_regions) * max(len(ids) * shape[0] * shape[1] for ids, shape in self.base_plan(self.epoch).batches)

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

    def shape_counts(self) -> dict[tuple[int, int, int], int]:
        """Batches per epoch of each (samples, height, width): which shapes the run spends its steps on."""
        counts: dict[tuple[int, int, int], int] = {}
        for ids, (height, width) in self.base_plan(self.epoch).batches:
            key = (len(ids) * self.noiser.max_regions, height, width)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def max_batch_size(self, regions: int | None = None) -> int:
        return (regions or self.noiser.max_regions) * max(len(ids) for ids, _ in self.base_plan(self.epoch).batches)

    def probe_batch(self, like: dict, count: int = 1) -> dict:
        """``count`` samples at the largest shape of the data, noised like a real batch, wearing the prompt of ``like``'s
        first sample: what a memory probe runs through the model."""
        ids, (height, width) = max(self.base_plan(self.epoch).batches, key=lambda b: b[1][0] * b[1][1])
        x0 = torch.zeros(count, int(self.latents.spec["channels"]), height, width)
        batch = self.noiser(x0, torch.Generator().manual_seed(self.seed))
        batch.update({"samples": [ids[i % len(ids)] for i in range(count)], "prompts": list(like["prompts"][:1]) * count, "epoch": self.epoch,
                      "prompts_version": like.get("prompts_version"), "sizes_px": [(height * self._factor, width * self._factor)] * count})
        return batch

    def __len__(self) -> int:
        """Batches of the current epoch (the same for every epoch unless a ramp starts the run with smaller batches)."""
        return len(self.plan_for(self.epoch))

    def __iter__(self) -> Iterator[dict]:
        epoch = self.epoch
        self.emit(EpochStarted(epoch))
        self.prompts.set_epoch(epoch)
        plan = self.plan_for(epoch)
        version, texts = self.prompts.snapshot()
        last = len(plan.batches) - 1
        offset, self._offset = self._offset, 0
        worker = Prefetch(lambda: (self._make(i, ids, shape, epoch, version, texts, i == last, self.batches_before(epoch))
                                   for i, (ids, shape) in enumerate(plan.batches) if i >= offset), self.depth)
        served = 0
        try:
            for batch in worker:
                served += len(batch["samples"]) // (int(batch["region"].max()) + 1)
                self.position += 1
                yield batch
        finally:
            worker.close()
        self.emit(EpochEnding(epoch, served))
        self.epoch = epoch + 1

    def _make(self, index: int, ids: tuple[str, ...], shape: tuple[int, int], epoch: int, version: int,
              texts: dict[str, str], last: bool = False, first: int = 0) -> dict:
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
        copies = self.noiser.regions_at(first + index) if len(ids) else 1
        if copies > 1:  # one copy of the images per region, region by region
            x0, valid, true_sizes, placed, ids = x0.repeat(copies, 1, 1, 1), valid.repeat(copies, 1, 1, 1), true_sizes * copies, placed * copies, ids * copies
        batch = self.noiser(x0, generator, copies)
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
