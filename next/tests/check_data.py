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
        from ..graph import ExecutionContext, Environment, SettingsRegistry
        from ..data.store import DatasetStore
        from ..nodes.dataset.batching import BatcherNode, BucketerNode, NoiserNode
        from ..nodes.dataset.images import DatasetLibraryNode
        from ..nodes.dataset.prompts import CaptionsNode, TagSwapNode

        store = DatasetStore(tmp / "store")
        lib = store.library("demo", create=True)
        lib.ingest(tmp / "images")
        lib.close()
        ctx = ExecutionContext(env=Environment({"datasets": store}))
        node = lambda cls: cls(ctx.scoped("n", cls.READS_SETTINGS, cls.__name__))  # noqa: E731
        got = node(DatasetLibraryNode).build(dataset="demo")["images"]
        check(len(got) == 6, "Dataset Library node returns the set")
        table = node(TagSwapNode).build(prompts=node(CaptionsNode).build(images=got, seed=1)["prompts"])["prompts"]
        check(len(table.stages) == 1, "stage node appends to the table")
        seeded = lambda base, uid: ExecutionContext(settings=SettingsRegistry().snapshot({"seed.base": base})).scoped(  # noqa: E731
            uid, ("seed.base",), "n")
        derive = lambda base, uid, value=None: CaptionsNode(seeded(base, uid)).derived_seed(value)  # noqa: E731
        check(derive(-1, "a") == 0 and derive(7, "a") == derive(7, "a") != derive(7, "b") != derive(8, "b") and derive(7, "a", 3) == 3,
              "a node seed left empty is 0 without a base, else derived from base and node id; an explicit seed wins")
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
        memory_pressure(tmp / "pressure")


class TightEncoder(FakeEncoder):
    """Runs out of memory for more than ``room`` images at a time, like a card that is nearly full."""

    def __init__(self, room: int):
        self.room, self.calls = room, []

    def encode(self, images):
        self.calls.append(len(images))
        if len(images) > self.room:
            raise torch.OutOfMemoryError("XPU out of memory")
        return super().encode(images)


def memory_pressure(tmp: Path) -> None:
    print("encoding under memory pressure")
    folder = tmp / "images"
    folder.mkdir(parents=True)
    make_images(folder)
    images = ImageLibrary(tmp / "lib.sqlite")
    images.ingest(folder)
    snapshot = images.snapshot()
    policy = SizePolicy(base=64, mode="fit", multiple=8)
    reference = LatentCache(tmp / "ref", "v", LATENTS)
    reference.build(snapshot, policy, FakeEncoder(), batch_size=4)
    cache, tight, recovered = LatentCache(tmp / "tight", "v", LATENTS), TightEncoder(room=1), []
    stats = cache.build(snapshot, policy, tight, batch_size=4, recover=lambda: recovered.append(1))
    got, want = cache.view(snapshot, policy), reference.view(snapshot, policy)
    same = got.samples == want.samples and all(torch.equal(got.get(s), want.get(s)) for s in got.samples)
    check(stats.encoded_images == 6 and same and recovered, "an out-of-memory encode is halved down to what fits, with the same latents")
    check(set(tight.calls[-4:]) == {1}, f"once a size is known to fit it is kept ({tight.calls})")
    raises(RuntimeError, lambda: LatentCache(tmp / "never", "v", LATENTS).build(snapshot, policy, TightEncoder(room=0), batch_size=2),
           "one image that never fits is an error that says so", "does not fit")


class LatentSpecLike:
    scale, spatial_factor, shift = 0.13025, 8, 0.0

    def __init__(self, channels: int):
        self.channels = channels


def shape_limit():
    from ..data.bucketing import bucket_map, padding_stats

    shapes = [(12 + 4 * i, 20 + 4 * j) for i in range(5) for j in range(5)] * 2 + [(12, 12)]
    free = bucket_map(shapes, 4)
    check(len(set(free.values())) == 25 + 1 and all(free[s] == s for s in shapes), "without a limit, shapes keep their own padded size")
    few = bucket_map(shapes, 4, 6)
    check(len(set(few.values())) == 6 and all(b[0] >= s[0] and b[1] >= s[1] for s, b in few.items()), "a limit merges shapes, and every sample still fits in its bucket")
    stats = padding_stats(shapes, 4, 6)
    check(stats["shapes_out"] == 6 and stats["pad_mean"] < 0.3, f"merging the cheapest pairs first keeps the padding small ({stats['pad_mean']:.0%})")
    check(bucket_map([(8, 8), (8, 8)], 4, 3) == {(8, 8): (8, 8)}, "a limit above the shape count changes nothing")


def batch_area_limit():
    import random

    from ..data.bucketing import make_plan

    def size_of(h: int, w: int, batch: int) -> int:
        plan = make_plan({f"a{i}": (h, w) for i in range(batch)}, batch_size=batch, full_batch_area=96 * 96, shuffle=False, keep_incomplete=True,
                         rng=random.Random(0))
        return len(plan.batches[0][0])

    check([size_of(96, 96, 4), size_of(96, 100, 4), size_of(96, 144, 4), size_of(96, 193, 4), size_of(96, 400, 4)] == [4, 3, 2, 1, 1],
          "a batch holds as many samples as fit the pixel budget of the full batch, rounded down, and at least one")
    check(size_of(96, 120, 2) == 1 and size_of(96, 96, 2) == 2 and size_of(96, 150, 3) == 1, "small batches follow the same rule")
    plain = make_plan({f"a{i}": (96, 400) for i in range(4)}, batch_size=4, shuffle=False, keep_incomplete=True, rng=random.Random(0))
    check(len(plain.batches[0][0]) == 4, "without a limit batches stay full")


def padding_fill():
    """Mirror padding reflects the noisy valid block into the padding, changes nothing inside it, and adds no information."""
    from ..data.padding import mirror_fill, mirror_index

    check(mirror_index(6, 1, 3).tolist() == [2, 1, 2, 3, 2, 1] and mirror_index(5, 1, 1).tolist() == [1] * 5,
          "positions outside the block reflect into it, and repeat when the padding is longer than the block")
    canvas = torch.arange(36.0).reshape(1, 6, 6)
    filled = mirror_fill(canvas, 1, 2, 3, 3)
    block = canvas[:, 1:4, 2:5]
    rows, cols = [2, 1, 2, 3, 2, 1], [4, 3, 2, 3, 4, 3]
    check(torch.equal(filled[:, 1:4, 2:5], block) and torch.equal(filled, canvas[:, rows][:, :, cols]),
          "the valid block is untouched and the rest is its reflection")
    tmp = Path(tempfile.mkdtemp())
    plain = list(build(tmp / "z", multiple=16, batch_size=3)[5])
    mirrored = list(build(tmp / "m", multiple=16, batch_size=3, fill="mirror")[5])
    same_inside = all(torch.equal(a["x_t"] * a["valid_mask"], b["x_t"] * b["valid_mask"]) for a, b in zip(plain, mirrored))
    differs_outside = any(not torch.equal(a["x_t"] * (1 - a["valid_mask"]), b["x_t"] * (1 - b["valid_mask"])) for a, b in zip(plain, mirrored))
    check(same_inside and differs_outside and all(torch.equal(a["target"], b["target"]) for a, b in zip(plain, mirrored)),
          "through the batcher, mirroring changes only the padded part of the input (valid part and target are the same)")
    raises(ValueError, lambda: build(tmp / "bad", fill="blur"), "an unknown fill is refused", "fill must be")


def packing():
    """Left-over samples fill short batches: into larger existing shapes, or into their bounding shape."""
    import random

    from ..data.bucketing import make_plan

    shapes = {f"a{i}": (96, 96) for i in range(6)}
    shapes.update({f"b{i}": (96, 144) for i in range(3)})
    shapes.update({f"c{i}": (144, 96) for i in range(3)})
    shapes.update({f"d{i}": (88, 96) for i in range(2)})

    def plan(pack, area=0, shapes=shapes, batch=4):
        return make_plan(shapes, batch_size=batch, full_batch_area=area, pack=pack, shuffle=True, keep_incomplete=True, rng=random.Random(1))

    off, on = plan(0.0), plan(0.3)
    used = sorted(s for ids, _ in on.batches for s in ids)
    check(len(off.batches) == 5 and len(on.batches) == 4 and used == sorted(shapes) and on.packed == 2,
          f"two 88x96 samples top up the short 96x96 batch: 5 batches become 4 ({[(len(i), s) for i, s in on.batches]})")
    fits = all(shapes[s][0] <= h and shapes[s][1] <= w and 1 - shapes[s][0] * shapes[s][1] / (h * w) <= 0.3 for ids, (h, w) in on.batches for s in ids)
    check(fits, "every sample fits its batch shape and no sample is more than 30% padding")
    check(plan(0.05).packed == 0 and len(plan(0.05).batches) == 5, "a sample whose padding would pass the limit stays where it was")
    two = {**{f"p{i}": (96, 128) for i in range(2)}, **{f"q{i}": (128, 96) for i in range(2)}}
    two = {**{f"p{i}": (96, 128) for i in range(2)}, **{f"q{i}": (128, 96) for i in range(2)}}
    check(plan(0.3, shapes=two).packed == 0, "leftovers that no existing shape holds together are not combined into a new shape")
    hosted = {**{f"h{i}": (128, 128) for i in range(4)}, "p": (96, 128), "q": (128, 96)}
    joined = plan(0.3, shapes=hosted)
    check(sorted((len(i), s) for i, s in joined.batches) == [(2, (128, 128)), (4, (128, 128))] and len(plan(0.0, shapes=hosted).batches) == 3
          and {s for _, s in joined.batches} <= {(128, 128)},
          "a portrait and a landscape leftover share the existing shape that holds both, and no new shape appears")
    check({s for _, s in plan(0.3).batches} <= {s for _, s in plan(0.0).batches},
          "packing never produces a shape the plan did not have without it")
    limited = plan(0.4, area=96 * 96 * 3 // 2, shapes={**{f"p{i}": (96, 144) for i in range(2)}, **{f"q{i}": (96, 192) for i in range(3)}})
    check(all(len(ids) <= max(1, 4 * (96 * 96 * 3 // 2) // (h * w)) for ids, (h, w) in limited.batches),
          "packed batches never exceed the size the batch-area limit allows for their shape")
    dropping = make_plan(shapes, batch_size=4, pack=0.3, shuffle=True, keep_incomplete=False, rng=random.Random(1))
    check(dropping.dropped == 6 and sum(len(i) for i, _ in dropping.batches) == len(shapes) - 6,
          "with short batches dropped, packing keeps the samples that fill a batch")


def prompt_lifetimes_and_store() -> None:
    """Prompts say how long they are worth keeping; only stable ones reach the disk store, and a second run reads them."""
    from ..data.conditioning_cache import ConditioningCache
    from ..data.conditioning_store import ConditioningStore
    from ..data.prompts import CaptionDropout, Lifetime, Prompt, PromptTable, TagShuffle, lifetime_of
    from .check_engine import FakeKit

    tmp = Path(tempfile.mkdtemp())
    base = {"a": "red, round, small", "b": "blue, square, big", "c": "green, flat, wide"}
    plain = PromptTable(base)
    plain.set_epoch(0)
    check(all(lifetime_of(t) is Lifetime.STABLE for t in plain.snapshot()[1].values()), "base captions are stable")
    shuffled = PromptTable(base, (TagShuffle(),), seed=3)
    shuffled.set_epoch(0)
    kinds = {t: lifetime_of(t) for t in shuffled.snapshot()[1].values()}
    check(any(k is Lifetime.FLEETING for k in kinds.values()) and all(isinstance(t, str) for t in kinds),
          "a shuffled caption is fleeting, and still an ordinary string")
    drop = PromptTable(base, (CaptionDropout(p=1.0, mode="empty"),))
    drop.set_epoch(0)
    check(all(lifetime_of(t) is Lifetime.STABLE for t in drop.snapshot()[1].values()), "dropout draws from a few results and stays stable")
    check(lifetime_of("plain str") is Lifetime.STABLE and Prompt("x", Lifetime.FLEETING) == "x", "a plain str counts as stable and a Prompt equals its text")

    kit = FakeKit()
    store = ConditioningStore(tmp / "cond", "ident", kit)
    first = ConditioningCache(kit, store=store)
    prompts = [Prompt("keep me"), Prompt("let go", Lifetime.FLEETING)]
    check(first.warm(prompts) == 2 and len(store) == 1, "only the stable prompt is written to disk")
    second = ConditioningCache(kit, store=ConditioningStore(tmp / "cond", "ident", kit))
    check(second.warm(prompts) == 1 and second.loaded == 1, "a later run reads the stable prompt and encodes only the fleeting one")
    a, b = first.get(["keep me"]), second.get(["keep me"])
    check(torch.equal(a.ctx, b.ctx) and torch.equal(a.y, b.y), "what comes back from disk is what was stored")
    other = ConditioningCache(kit, store=ConditioningStore(tmp / "cond", "other-encoder", kit))
    check(other.warm(prompts) == 2, "another encoder identity never reads these entries")
    small = ConditioningStore(tmp / "small", "ident", kit, max_bytes=1)
    ConditioningCache(kit, store=small).warm(prompts)
    check(small.full and len(small) == 0, "a full store stops writing and says so; the run continues from memory")


if __name__ == "__main__":
    run(main, shape_limit, batch_area_limit, packing, padding_fill, prompt_lifetimes_and_store)
