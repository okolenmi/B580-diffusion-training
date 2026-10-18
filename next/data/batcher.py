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
from .bucketing import Plan, make_plan, padding_stats
from .latent_cache import LatentSet
from .noise import Noiser
from .prefetch import Prefetch
from .prompts import PromptTable
from .ids import image_of


class Batcher:
    def __init__(self, latents: LatentSet, prompts: PromptTable, noiser: Noiser, *, batch_size: int = 1,
                 shuffle: bool = True, keep_incomplete: bool = False, multiple: int = 1, max_shapes: int = 0, seed: int = 0,
                 depth: int = 2, pin: bool = False, start_epoch: int = 0,
                 emit: Callable[[EventType], None] | None = None):
        orphans = [s for s in latents.samples if not prompts.knows(image_of(s))]
        if orphans:
            raise ValueError(f"{len(orphans)} sample(s) have latents but no prompt entry (first: {orphans[0]}); "
                             f"the prompt table must be built from the same images")
        self.latents, self.prompts, self.noiser = latents, prompts, noiser
        self.batch_size, self.shuffle, self.keep_incomplete = batch_size, shuffle, keep_incomplete
        self.multiple, self.max_shapes, self.seed, self.depth, self.pin = multiple, max_shapes, seed, depth, pin
        self.epoch = start_epoch
        self.emit = emit or (lambda event: None)
        self._factor = int(latents.spec["spatial_factor"])
        self.plan_for(start_epoch)  # an impossible configuration fails now, not on the first step

    def plan_for(self, epoch: int) -> Plan:
        return make_plan(self.latents.shapes, batch_size=self.batch_size, multiple=self.multiple, max_shapes=self.max_shapes,
                         shuffle=self.shuffle, keep_incomplete=self.keep_incomplete,
                         rng=random.Random(f"{self.seed}:plan:{epoch}"))

    def shape_summary(self) -> dict:
        """What the model will see: distinct input shapes (batch size included, as a short batch is a shape of its own)
        and how much of the bucketed area is padding."""
        plan = self.plan_for(self.epoch)
        stats = padding_stats(list(self.latents.shapes.values()), self.multiple, self.max_shapes)
        return {"samples": stats["samples"], "shapes_in": stats["shapes_in"], "bucket_shapes": stats["shapes_out"],
                "batch_shapes": len({(len(ids), *shape) for ids, shape in plan.batches}), "batches": len(plan),
                "padding_mean": round(stats["pad_mean"], 3), "padding_worst": round(stats["pad_max"], 3)}

    def probe_batch(self, like: dict) -> dict:
        """One sample at the largest shape of the data, noised like a real batch, wearing the prompt of ``like``'s
        first sample: what a memory probe runs through the model."""
        ids, (height, width) = max(self.plan_for(self.epoch).batches, key=lambda b: b[1][0] * b[1][1])
        x0 = torch.zeros(1, int(self.latents.spec["channels"]), height, width)
        batch = self.noiser(x0, torch.Generator().manual_seed(self.seed))
        batch.update({"samples": [ids[0]], "prompts": list(like["prompts"][:1]), "epoch": self.epoch,
                      "prompts_version": like.get("prompts_version"), "sizes_px": [(height * self._factor, width * self._factor)]})
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
        worker = Prefetch(lambda: (self._make(i, ids, shape, epoch, version, texts)
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
              texts: dict[str, str]) -> dict:
        rng = random.Random(f"{self.seed}:batch:{epoch}:{index}")
        generator = torch.Generator().manual_seed(rng.getrandbits(62))
        height, width = shape
        x0 = torch.zeros(len(ids), int(self.latents.spec["channels"]), height, width)
        valid = torch.zeros(len(ids), 1, height, width)
        true_sizes = []
        for k, sample in enumerate(ids):
            latent = self.latents.get(sample)
            h, w = latent.shape[-2:]
            top = rng.randint(0, height - h) if height > h else 0
            left = rng.randint(0, width - w) if width > w else 0
            x0[k, :, top:top + h, left:left + w] = latent
            true_sizes.append((h * self._factor, w * self._factor))
            valid[k, :, top:top + h, left:left + w] = 1.0
        batch = self.noiser(x0, generator)
        batch.update({
            "samples": list(ids),
            "prompts": [texts[image_of(s)] for s in ids],
            "epoch": epoch,
            "prompts_version": version,
            "sizes_px": true_sizes,
        })
        if self.multiple > 1 and bool((valid == 0).any()):
            batch["valid_mask"] = valid
        if self.pin:
            batch = {k: v.pin_memory() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch
