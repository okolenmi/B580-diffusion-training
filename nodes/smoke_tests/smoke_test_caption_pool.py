"""Caption pool: a prompt per image built from the dataset's real captions.

Logic-level, CPU, on a tiny synthetic dataset written to a temp directory
(real metadata.db + real safetensors shard, so the loader runs end to end).
Pins: the prompts come only from the dataset's own captions; n stays within
n_min..n_max; '.'/',' divide as documented; a seeded pool is reproducible;
`variants` bounds the prompt set and `universe()` names exactly it; fresh mode
(variants=0) has no universe; the pool needs bucketing and two captions; a
batch's `prompt` and `prompts` agree; the prewarm discovery sees the
mixed-caption keys and the universe. Whether the varied prompts help a real
LoRA is a render question this does not answer.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from manager.caption_pool import CaptionPool
from manager.db import add_shard, add_source, add_trajectory, init_local_db
from manager.loader import ManagedDatasetLoader
from manager.storage import ShardWriter

# pin_memory needs an accelerator driver; the batches are what is under test.
ManagedDatasetLoader._pin_batch = staticmethod(lambda batch: batch)

CAPTIONS = ["a red fox, snowy field.", "blue boat, calm sea", "old castle on a hill.",
            "green frog, wet leaf", "tall pine, morning fog"]


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def make_dataset(root: Path, captions) -> Path:
    ds = root / "ds"
    (ds / "shards").mkdir(parents=True)
    db = ds / "metadata.db"
    init_local_db(db)
    src = add_source(db, "s", "real")
    writer = ShardWriter(ds / "shards" / "x0.safetensors")
    idx = [writer.add_image_latent(torch.randn(1, 4, 8, 8)) for _ in captions]
    count, size = writer.write()
    shard = add_shard(db, "shards/x0.safetensors", count, size)
    for i, (cap, k) in enumerate(zip(captions, idx)):
        add_trajectory(db, src, shard, k, 1, i, cap, latent_h=8, latent_w=8)
    return ds


def check_prompts_come_from_the_real_captions():
    print("[captions mode: every part of a prompt is a real caption; n in range; own first]")
    pool = CaptionPool(n_min=2, n_max=3, mode="captions", variants=6, seed=1)
    check(pool.prepare(CAPTIONS), "pool refused a 5-caption dataset")
    trimmed = {c.rstrip(". ,") for c in CAPTIONS}
    for i, own in enumerate(CAPTIONS):
        for _ in range(20):
            prompt = pool.prompt_for(i)
            pieces = prompt.split(". ")
            check(2 <= len(pieces) <= 3, (len(pieces), prompt))
            check(all(p in trimmed for p in pieces), prompt)
            check(pieces[0] == own.rstrip(". ,"), (own, prompt))
    print("    PASS")


def check_parts_mode_splits_on_dividers():
    print("[parts mode: '.' and ',' split captions into tags; no duplicates; only real tags]")
    pool = CaptionPool(n_min=3, n_max=3, mode="parts", variants=5, seed=2)
    pool.prepare(CAPTIONS)
    tags = {t.strip() for c in CAPTIONS for t in c.replace(".", ",").split(",") if t.strip()}
    for i in range(len(CAPTIONS)):
        for _ in range(10):
            parts = pool.prompt_for(i).split(", ")
            check(len(parts) == len({p.lower() for p in parts}), parts)
            check(set(parts) <= tags, set(parts) - tags)
    print("    PASS")


def check_seeded_pool_is_reproducible_and_bounded():
    print("[seed reproduces; variants bounds the set; universe() is exactly that set]")
    a = CaptionPool(variants=3, seed=7); a.prepare(CAPTIONS)
    b = CaptionPool(variants=3, seed=7); b.prepare(CAPTIONS)
    check(a.universe() == b.universe(), "same seed, different prompt sets")
    check(len(a.universe()) <= 3 * len(CAPTIONS), len(a.universe()))
    served = {a.prompt_for(i) for i in range(len(CAPTIONS)) for _ in range(40)}
    check(served == set(a.universe()), "served prompts differ from universe()")
    fresh = CaptionPool(variants=0, seed=7); fresh.prepare(CAPTIONS)
    check(fresh.universe() == [], "fresh mode claimed an enumerable set")
    check(len({fresh.prompt_for(0) for _ in range(60)}) > 3, "fresh mode did not vary")
    print("    PASS")


def check_a_pool_without_variety_is_a_no_op():
    print("[<2 distinct non-empty captions: inactive, images keep their own caption]")
    for caps in (["", "", ""], ["same", "same"], ["only", ""]):
        pool = CaptionPool(); ok = pool.prepare(caps)
        check(not ok and not pool.active, caps)
        check([pool.prompt_for(i) for i in range(len(caps))] == caps, caps)
        check(pool.universe() == [], caps)
    print("    PASS")


def check_empty_own_caption_still_gets_a_prompt():
    print("[include_own with an empty own caption: prompt built from others only]")
    pool = CaptionPool(n_min=2, n_max=2, variants=3, seed=3)
    pool.prepare(["", *CAPTIONS])
    for _ in range(10):
        p = pool.prompt_for(0)
        check(p and len(p.split(". ")) == 2, p)
    print("    PASS")


def check_bad_settings_are_refused():
    print("[bad settings raise ValueError]")
    for kw in ({"n_min": 0}, {"n_min": 3, "n_max": 2}, {"mode": "x"},
               {"variants": -1}, {"dividers": ""}):
        try:
            CaptionPool(**kw)
        except ValueError:
            continue
        check(False, f"{kw} accepted")
    print("    PASS")


def check_the_loader_applies_the_pool():
    print("[loader end to end: needs bucketing; batches carry pool prompts; prompt == prompts[0]]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make_dataset(Path(tmp), CAPTIONS)
        try:
            ManagedDatasetLoader(ds, batch_size=2, caption_pool=CaptionPool())
        except ValueError:
            pass
        else:
            check(False, "pool without bucketing was accepted")
        pool = CaptionPool(n_min=2, n_max=3, variants=4, seed=5)
        loader = ManagedDatasetLoader(ds, batch_size=5, shuffle=False,
                                      shape_bucket_multiple=8, keep_incomplete=True,
                                      group_by_size_only=True, caption_pool=pool)
        batches = list(loader)
        check(batches, "no batches")
        trimmed = {c.rstrip(". ,") for c in CAPTIONS}
        for b in batches:
            check(b["prompt"] == b["prompts"][0], "prompt != prompts[0]")
            check(len(b["prompts"]) == b["x_t"].shape[0], "prompts/x_t length")
            for p in b["prompts"]:
                check(all(piece in trimmed for piece in p.split(". ")), p)
        universe = loader.prompt_universe()
        check(universe and set(p for b in batches for p in b["prompts"]) <= set(universe),
              "served a prompt outside universe()")
        from nodes.dataset.managed import ManagedDatasetBatchSource
        check(ManagedDatasetBatchSource(loader).prompt_universe() == universe, "source passthrough")
    print("    PASS")


def check_len_before_iteration_still_builds_the_pools():
    print("[len() before iter(): caption pool and dropout pool are built (they were not)]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make_dataset(Path(tmp), CAPTIONS)
        loader = ManagedDatasetLoader(ds, batch_size=1, shuffle=False,
                                      caption_dropout=0.5)
        len(loader)
        check(len(loader._caption_pool) == len(CAPTIONS), loader._caption_pool)
        pool = CaptionPool(variants=2, seed=1)
        loader = ManagedDatasetLoader(ds, batch_size=1, shape_bucket_multiple=8,
                                      caption_pool=pool)
        len(loader)
        check(pool.active, "pool inactive after len()")
    print("    PASS")


def check_no_pool_changes_nothing():
    print("[no pool: prompts are the stored captions, universe empty]")
    with tempfile.TemporaryDirectory() as tmp:
        ds = make_dataset(Path(tmp), CAPTIONS)
        loader = ManagedDatasetLoader(ds, batch_size=1, shuffle=False)
        got = sorted(b["prompt"] for b in loader)
        check(got == sorted(CAPTIONS), got)
        check(loader.prompt_universe() == [], "universe without a pool")
    print("    PASS")


def check_prewarm_discovery_sees_mixed_captions_and_the_universe():
    print("[prewarm discovery: mixed batch -> (prompt,1) keys; universe added; blank added]")
    from nodes.model.text_encoder_prewarm import discover_dataset_keys
    from nodes.train.conditioning_blend import BLANK_PROMPT, with_blank_prompt
    with tempfile.TemporaryDirectory() as tmp:
        ds = make_dataset(Path(tmp), CAPTIONS)
        pool = CaptionPool(n_min=2, n_max=3, variants=4, seed=9)
        loader = ManagedDatasetLoader(ds, batch_size=5, shuffle=False,
                                      shape_bucket_multiple=8, keep_incomplete=True,
                                      group_by_size_only=True, caption_pool=pool)
        from nodes.dataset.managed import ManagedDatasetBatchSource
        disc = discover_dataset_keys(ManagedDatasetBatchSource(loader))
        prompts = {k[0] for k in disc.keys}
        check(set(loader.prompt_universe()) <= prompts,
              "universe prompts missing from the warm set")
        check(all(k[1] == 1 for k in disc.keys), "mixed prompts must key at batch size 1")
        check(all(len(k) == 5 for k in disc.keys), "key shape")
        check(BLANK_PROMPT not in prompts, "blank present before it was asked for")
        with_blank = with_blank_prompt(disc)
        check((BLANK_PROMPT, 1) in {(k[0], k[1]) for k in with_blank.keys}, "blank not added")
        check(with_blank_prompt(with_blank).keys == with_blank.keys, "not idempotent")
    print("    PASS")


def check_the_node():
    print("[CaptionPoolNode: defaults, seed -1 = None, bad settings fail at build]")
    from nodes.dataset.caption_pool import CaptionPoolNode, CaptionPoolSettings
    out = CaptionPoolNode().build()["pool"]
    check(isinstance(out, CaptionPoolSettings) and out.seed is None, out)
    check(CaptionPoolNode().build(seed=4)["pool"].seed == 4, "seed")
    try:
        CaptionPoolNode().build(n_min=5, n_max=2)
    except ValueError:
        pass
    else:
        check(False, "bad n range accepted")
    from nodes.dataset.managed import ManagedDatasetSourceNode
    check("caption_pool" in ManagedDatasetSourceNode.INPUTS, "port missing")
    check(ManagedDatasetSourceNode.INPUTS["caption_pool"].default is None, "default")
    print("    PASS")


def main():
    check_prompts_come_from_the_real_captions()
    check_parts_mode_splits_on_dividers()
    check_seeded_pool_is_reproducible_and_bounded()
    check_a_pool_without_variety_is_a_no_op()
    check_empty_own_caption_still_gets_a_prompt()
    check_bad_settings_are_refused()
    check_the_loader_applies_the_pool()
    check_len_before_iteration_still_builds_the_pools()
    check_no_pool_changes_nothing()
    check_prewarm_discovery_sees_mixed_captions_and_the_universe()
    check_the_node()
    print("\nSMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
