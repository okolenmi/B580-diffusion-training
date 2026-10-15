"""Data pipeline contract: what must keep holding when the library, caches, prompts or batcher change.

    python -m next.tests.check_data

Covers the epoch barrier and its event order, per-VAE cache refusal, prompt determinism across epochs
(including resume), and the join between latents and prompts, on synthetic images with a fake encoder.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from PIL import Image

from ..data.batcher import Batcher
from ..data.ids import vae_identity
from ..data.images import SizePolicy
from ..data.latent_cache import CacheMismatch, LatentCache
from ..data.library import ImageLibrary, report
from ..data.noise import Noiser, TimestepPolicy
from ..data.prompts import CaptionDropout, PromptTable, TagShuffle, TagSwap
from ..families.api.header import read_header
from ..families.sdxl.latents import LATENTS
from ..families.sdxl.objective import SDXLObjective
from ..graph.events import EpochEnding, EpochStarted, PromptsChanged
from .support import check, raises, run

TAGS = ["a girl", "red hair", "smiling", "outdoors", "sunset", "hat"]


class FakeEncoder:
    """Stands in for the VAE codec: one latent value per 8x8 block, so shapes follow the real ones."""

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.avg_pool2d(images.float() / 255.0, 8)
        return pooled[:, :1].repeat(1, 4, 1, 1)


def make_images(root: Path) -> None:
    for i in range(6):
        size = (96, 64) if i % 2 == 0 else (64, 96)
        Image.new("RGB", size, (i * 40, 10, 200)).save(root / f"img{i}.png")
        (root / f"img{i}.txt").write_text(", ".join(TAGS[: 3 + i % 3] + [f"tag{i}"]))


def build(tmp: Path, **batcher_kwargs):
    folder = tmp / "images"
    folder.mkdir(parents=True, exist_ok=True)
    make_images(folder)
    library = ImageLibrary(tmp / "lib.sqlite")
    result = library.ingest(folder)
    images = library.snapshot()
    cache = LatentCache(tmp / "caches", "vae0", LATENTS, label="test-vae")
    policy = SizePolicy(base=64, mode="fit", multiple=8)
    stats = cache.build(images, policy, FakeEncoder(), batch_size=2)
    latents = cache.view(images, policy)
    events: list = []
    prompts = PromptTable.from_images(images, seed=3).with_stage(TagSwap("random", keep_first=1))
    prompts.listener = events.append
    kwargs = dict(batch_size=2, shuffle=True, keep_incomplete=True, seed=5)
    kwargs.update(batcher_kwargs)
    batcher = Batcher(latents, prompts, Noiser(SDXLObjective()), emit=events.append, **kwargs)
    return library, images, cache, latents, prompts, batcher, events, result, stats


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        library, images, cache, latents, prompts, batcher, events, result, stats = build(tmp)

        print("library and cache")
        check(result.added == 6 and not result.skipped, "six images ingested with sidecar captions")
        again = library.ingest(tmp / "images")
        check(again.added == 0 and again.known == 6, "re-ingest keeps content-hash ids")
        check(len(report(images)["empty_captions"]) == 0, "report finds no empty captions")
        check(stats.encoded_images == 6 and len(latents) == 6, "every image encoded once")
        check(not cache.missing(images, SizePolicy(base=64, mode="fit", multiple=8)), "second build has nothing to do")
        raises(LookupError, lambda: cache.view(images, SizePolicy(base=128, mode="fit", multiple=8)),
               "a policy that was never built is refused", "build the cache first")
        other = LatentSpecLike(channels=16)
        raises(CacheMismatch, lambda: LatentCache(tmp / "caches", "vae0", other), "cache refuses another latent space",
               "graph's VAE")

        print("epoch barrier")
        order = []
        batcher.emit = lambda e: (order.append(type(e).__name__), events.append(e))
        seen = []
        for batch in batcher:
            order.append("batch")
            seen.append(batch["samples"])
        check(order[0] == "EpochStarted" and order[-1] == "EpochEnding", "events frame the epoch")
        check("PromptsChanged" not in order[1:-1], "prompts do not change mid-epoch")
        check(sorted(s for b in seen for s in b) == sorted(latents.samples), "each sample served once")
        check(batcher.epoch == 1, "epoch advances after the last batch")
        ending = [e for e in events if isinstance(e, EpochEnding)][-1]
        check(ending.samples == 6, "EpochEnding counts served samples")

        print("prompts")
        first = dict(prompts.snapshot()[1])
        started = [e for e in events if isinstance(e, PromptsChanged)]
        check(bool(started), "first epoch announces PromptsChanged")
        list(batcher)
        second = dict(prompts.snapshot()[1])
        check(first != second, "tag swap picks different tags in the next epoch")
        resumed = PromptTable.from_images(images, seed=3).with_stage(TagSwap("random", keep_first=1))
        resumed.set_epoch(1)
        check(resumed.snapshot()[1] == second, "resuming at epoch 1 reproduces the same prompts")
        shuffled = PromptTable.from_images(images, seed=3).with_stage(TagShuffle(keep_first=1))
        shuffled.set_epoch(0)
        check(all(p.split(", ")[0] == "a girl" for p in shuffled.snapshot()[1].values()), "shuffle keeps leading tags")
        dropout = PromptTable.from_images(images, seed=3).with_stage(CaptionDropout(1.0, "empty"))
        dropout.set_epoch(0)
        check(set(dropout.snapshot()[1].values()) == {""}, "dropout with p=1 empties every prompt")

        print("batch contents")
        batch = next(iter(batcher))
        check(batch["x_t"].shape == batch["target"].shape and batch["x_t"].shape[1] == 4, "x_t and target share a shape")
        check(len(batch["prompts"]) == batch["x_t"].shape[0] == len(batch["t"]), "one prompt and one t per sample")
        check(batch["prompts_version"] >= 1 and "sigma" in batch, "batch carries prompts version and sigma")

        print("shape bucketing")
        _, _, _, latents2, _, padded, _, _, _ = build(tmp / "b", multiple=16, batch_size=3)
        padded2 = list(padded)
        shapes = {tuple(b["x_t"].shape[-2:]) for b in padded2}
        check(shapes == {(16, 16)} and all("valid_mask" in b for b in padded2), "mixed latent shapes bucket to one 16x16 shape with a valid mask")
        raises(ValueError, lambda: Batcher(latents2, PromptTable({}), Noiser(SDXLObjective())),
               "latents without prompts are refused at construction", "no prompt entry")

        print("nodes")
        from ..families.api.parts import PartInfo, PartKind, make_part
        from ..graph import ExecutionContext, Environment
        from ..data.store import DatasetStore
        from ..nodes.dataset.batching import BatcherNode, BucketerNode, NoiserNode
        from ..nodes.dataset.images import DatasetLibraryNode
        from ..nodes.dataset.prompts import CaptionsNode, TagSwapNode

        store = DatasetStore(tmp / "store")
        lib = store.library("demo", create=True)
        lib.ingest(tmp / "images")
        lib.close()
        ctx = ExecutionContext(env=Environment({"datasets": store}))
        node = lambda cls: cls(ctx.scoped("n", (), cls.__name__))  # noqa: E731
        got = node(DatasetLibraryNode).build(dataset="demo")["images"]
        check(len(got) == 6, "Dataset Library node returns the set")
        table = node(TagSwapNode).build(prompts=node(CaptionsNode).build(images=got, seed=1)["prompts"])["prompts"]
        check(len(table.stages) == 1, "stage node appends to the table")
        info = PartInfo(PartKind.DENOISER, "denoiser", "", 1, "F16", 1)
        noiser = node(NoiserNode).build(denoiser=make_part("sdxl", "x.safetensors", info), parameterization="v")["noiser"]
        buckets = node(BucketerNode).build(latents=latents, multiple=1)["buckets"]
        fed = node(BatcherNode).build(latents=latents, prompts=table, noiser=noiser, buckets=buckets, batch_size=2)["batches"]
        check(len(list(fed)) == len(fed), "Batcher node serves one epoch of the planned length")

        print("vae identity")
        from safetensors.torch import save_file
        a, b = tmp / "a.safetensors", tmp / "b.safetensors"
        save_file({"first_stage_model.w": torch.zeros(4, 4)}, str(a))
        save_file({"first_stage_model.w": torch.ones(4, 4)}, str(b))
        check(vae_identity(read_header(a), "first_stage_model.", LATENTS)
              != vae_identity(read_header(b), "first_stage_model.", LATENTS),
              "same layout, different weights: different identity")


class LatentSpecLike:
    scale, spatial_factor, shift = 0.13025, 8, 0.0

    def __init__(self, channels: int):
        self.channels = channels


if __name__ == "__main__":
    run(main)
