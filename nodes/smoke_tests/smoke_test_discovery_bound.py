"""Bounding the discovery pass, and reporting when it stopped early.

The discovery pass reads every latent off disk to read its shape -- 1.24 ms
per sample on the B580, measured -- to collect keys that cost 30 ms each to
warm. On a dataset with few distinct prompts that is almost all cost and no
benefit: at 1M samples it is **21 minutes to warm one prompt**, and all three
of this project's real datasets have exactly one.

So the pass stops after N consecutive batches introducing no new prompt, and
returns *why* it stopped. Not a bare set, because "found every key" and
"stopped looking" are the same value with very different consequences.

Every source here is a fake, deliberately: the question is what the counting
does, and a real shard file would only make it slower to answer. The shapes
are real ones -- (4, 128, 128) and (4, 64, 93) are measured latents from
`1024 aes` and `non-square`.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import random

import torch

from nodes.model.text_encoder_prewarm import (
    STOP_AFTER_NO_NEW_PROMPTS,
    StopReason,
    discover_dataset_keys,
)

CHECKS = 0


def check(condition: bool, message: str) -> None:
    global CHECKS
    CHECKS += 1
    if not condition:
        raise AssertionError(message)
    print(f"    PASS: {message}")


def batch(prompt: str, latent=(4, 128, 128), batch_size: int = 2):
    """One training batch, shaped as the real pipeline hands them over."""
    return {
        "x_t": torch.zeros(batch_size, *latent),
        "target": torch.zeros(batch_size, *latent),
        "t": 0.5,
        "prompt": prompt,
        "neg_prompt": "",
        "seed": 0,
        "metadata": {},
        "traj_type": "t",
    }


def source(n: int, prompts, latent=(4, 128, 128)):
    """`n` batches cycling `prompts`, so order is controlled exactly."""
    return [batch(prompts[i % len(prompts)], latent) for i in range(n)]


def source_from_prompts(prompts, latent=(4, 128, 128)):
    """One batch per entry, in exactly the order given."""
    return [batch(prompt, latent) for prompt in prompts]


def main() -> None:
    print("[a few prompts over a long dataset: stop early]")
    # What every real dataset here looks like: one prompt, many batches.
    d = discover_dataset_keys(source(1000, ["only one"]), max_batches=10_000)
    check(d.reason is StopReason.NO_NEW_PROMPTS,
          f"a thousand batches of one prompt stops early (reason "
          f"{d.reason.value})")
    check(d.batches_seen == STOP_AFTER_NO_NEW_PROMPTS + 1,
          f"after exactly N+1 batches, not 1,000 (saw {d.batches_seen}) -- "
          f"that is 1000x less disk for the same one prompt")
    check(d.prompts_found == 1 and len(d.keys) == 1,
          "having found the one prompt there was to find")
    check(d.truncated, "and says so rather than returning a set that looks "
          "complete")

    print("\n[many prompts, shuffled: the counter keeps resetting]")
    # This is the case the bound must not break. A shuffled loader puts a new
    # prompt every few batches, so 64 consecutive stale batches should not
    # happen -- and if it does, that is 200 prompts of information genuinely
    # missed, so it had better not.
    prompts = [f"caption number {i}" for i in range(200)]
    shuffled = [prompts[i % len(prompts)] for i in range(1000)]
    random.Random(20261004).shuffle(shuffled)
    d = discover_dataset_keys(source_from_prompts(shuffled), max_batches=10_000)
    check(d.reason is StopReason.COMPLETED,
          f"200 prompts shuffled through 1,000 batches never trips the "
          f"counter (reason {d.reason.value})")
    check(not d.truncated, "so it is not reported as truncated")
    check(d.batches_seen == 1000, f"and the whole dataset was read "
                                  f"({d.batches_seen} batches)")
    check(d.prompts_found == 200, f"finding all 200 prompts "
                                  f"(got {d.prompts_found})")

    print("\n[clustered prompts: scaling makes this safe too]")
    # An unshuffled source sorted by caption puts every prompt in one block,
    # which a *fixed* threshold trips -- it sees a long stale run right after
    # the first block. I had documented that as the failure mode. It is not,
    # and the test that showed it was the reason: the threshold scales with
    # prompts found, so by the time a clustered source stalls it has already
    # seen nearly all of them and needs a correspondingly long run to trip.
    # So both orderings complete and find everything, which is worth pinning
    # rather than leaving as a comment.
    clustered = [f"caption number {i // 5}" for i in range(1000)]
    d = discover_dataset_keys(source_from_prompts(clustered),
                              max_batches=10_000)
    check(d.reason is StopReason.COMPLETED,
          f"a clustered source also completes (reason {d.reason.value}) -- "
          f"scaling removed the failure mode a fixed threshold had")
    check(d.prompts_found == 200,
          f"finding all 200 of its prompts too (got {d.prompts_found})")

    print("\n[and a genuinely long stale tail still stops it]")
    # The bound has to be able to fire, or the tests above prove nothing about
    # it. A tail far longer than any plausible prompt gap is one: one prompt,
    # then a very long run of batches introducing nothing.
    tail = ["only one"] * 40 + ["late arrival"] + ["only one"] * 5000
    d = discover_dataset_keys(source_from_prompts(tail), max_batches=10_000)
    check(d.reason is StopReason.NO_NEW_PROMPTS,
          f"a 5,000-batch tail after the last new prompt stops it (reason "
          f"{d.reason.value})")
    check(d.prompts_found == 2,
          f"having found the 2 prompts there were ({d.prompts_found})")

    print("\n[new resolutions DO reset the counter -- a correction]")
    # I wrote this the other way round and asserted it. The reasoning was that
    # a resolution key is cheap to warm (1.6 ms against a prompt's 30 ms), so
    # a new one is not new information worth more disk. Measured on the real
    # data that is wrong: `non-square` has 44 resolutions over 121 batches, so
    # stopping on prompts alone at batch 65 missed 43 of them -- and a *missed*
    # key costs ~790 ms, because the miss self-loads CLIP. Cheap to warm and
    # cheap to miss are different properties and only the first one is true.
    shapes = [(4, 64, 60 + 2 * i) for i in range(8)]
    many_shapes = [batch("only one", latent=shapes[i % 8]) for i in range(1000)]
    d = discover_dataset_keys(many_shapes, max_batches=10_000)
    check(len(d.keys) == len(shapes),
          f"every resolution is collected before the bound can trip "
          f"({len(d.keys)} of {len(shapes)}) -- the counter is on whole keys")
    check(d.prompts_found == 1,
          "while still counting a single prompt, which is what makes the "
          "single-shape datasets stop early")

    print("\n[the cap still bounds an endless source]")
    endless = source(50, [f"p{i}" for i in range(50)])
    d = discover_dataset_keys(itertools_cycle(endless), max_batches=20)
    check(d.reason is StopReason.MAX_BATCHES,
          f"a source that never ends hits the cap, not the prompt counter "
          f"(reason {d.reason.value})")
    check(d.batches_seen == 21, f"at N+1 batches again ({d.batches_seen})")

    print("\n[the bound can be turned off, for a dataset that needs it]")
    d = discover_dataset_keys(source(1000, ["only one"]), max_batches=10_000,
                              stop_after_no_new_prompts=None)
    check(d.reason is StopReason.COMPLETED and d.batches_seen == 1000,
          f"stop_after_no_new_prompts=None reads everything ({d.batches_seen} "
          f"batches) -- the escape hatch for an unshuffled source whose "
          f"prompts are clustered")

    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {CHECKS} CHECKS PASSED")


def itertools_cycle(items):
    """An endless iterator over `items`, for the cap test."""
    while True:
        yield from items


if __name__ == "__main__":
    main()
