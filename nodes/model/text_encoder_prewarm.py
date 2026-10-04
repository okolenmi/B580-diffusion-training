"""PrewarmedTextEncoderNode: encode every (prompt, batch_size, height,
width) combination a dataset will actually use, once, then unload the
underlying text encoder entirely -- mirroring what core/trainer.py's own
pipeline already does (build a prompt cache, then `student_encoder.unload()`),
which the nodes/ pipeline didn't have yet. On an SDXL setup that's
roughly 1.5GB of CLIP-L+CLIP-G weights (fp16) sitting in VRAM for the
entire training run for no reason once the cache is warm -- likely the
single largest fixable difference between the two pipelines' VRAM use.

Why not just make CachingTextEncoderNode (nodes/model/text_encoder_cache.py)
do this reactively -- warm as it goes, unload once nothing new shows up?
There's no reliable signal for "nothing new is ever coming" in a
streaming decorator; it would have to guess. This node sidesteps
guessing by taking the dataset as an input and doing one real pass over
it first -- the exact (prompt, batch_size, height, width) keys
nodes/train/step_pipeline.py's EncodeConditioningPhase will request are
derived the same way it derives them (batch["x_t"].shape gives batch_size
and, *8 for the VAE downsample factor, height/width), so this doesn't
need to guess at that either. That pass is cheap: ManagedDatasetLoader-backed
sources just read already-stored tensors, no VAE/CLIP calls, and it also
warms the loader's own internal sample cache as a side effect.

Composes CachingTextEncoder (nodes/model/text_encoder_cache.py) rather
than being its own cache implementation -- this node's only real job is
figuring out *which* keys to warm and calling unload() once they're all
in, not re-solving caching. Both halves are module-level below:
`discover_dataset_keys()` does the pass, `warm_and_unload()` the warm
+ unload -- shared with the Resources Controller route's only
equivalent entry point, ManagedLoRATrainerNode's `prewarm_text_encoder`
Port (nodes/train/managed.py): that route never exposes trainer.clip
as a graph port, so this node's `encoder` input is unwirable there, and
the trainer node (the one place holding both trainer.clip and the
training batches) does the same job instead.

If something outside the discovered set is ever requested anyway (the
dataset changes between this node running and training starting, or any
other mismatch), this degrades, not breaks: the underlying encoder is
only moved to CPU by unload(), not destroyed, so a genuine cache miss
still returns a correct answer -- recomputed wherever the encoder
actually is at that moment: on the accelerator if a resource_control
handle is bound (the managed route's `prewarm_text_encoder` binds one,
so a miss self-loads and costs a slow-but-on-device encode), on CPU
otherwise (no handle, encoder parked on CPU -- the "recomputed on CPU
instead of the accelerator" case this docstring has always described).
Not something
to be usually relied on, but not a hard failure mode either.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from ..core import Port
from ..dataset.handle import TrainingBatchSource
from .text_encoder import TextEncoder, TextEncoderNode
from .text_encoder_cache import CachingTextEncoder

logger = logging.getLogger(__name__)

#: How many batches `discover_dataset_keys` will pull before giving up on
#: the source being finite. Far above any real dataset -- the largest here
#: is 152 batches -- so this is a backstop against an endless source, not
#: a budget. See `discover_dataset_keys`'s docstring for why truncating
#: costs time rather than correctness.
MAX_DISCOVERY_BATCHES = 100_000

#: Host-RAM budget the warm pass is measured against, in bytes. 8 GiB.
#:
#: This exists to answer "is this dataset too big to warm all at once",
#: with the arithmetic rather than with an opinion. Warming is O(distinct
#: prompts) in both time and RAM, and both are measured: 30.6 ms and
#: 621 KB per distinct prompt on the B580's XPU (1,505 ms on CPU), plus
#: 1.24 ms per dataset sample for the discovery pass itself, which reads
#: every latent off disk to read its shape.
#:
#: So 8 GiB is about **13,000 distinct prompts** -- 6.6 minutes of warm
#: pass. Past that the numbers say the current design is the wrong one, and
#: they say it while it is still cheap to find out rather than after the
#: allocation. A windowed design (prewarm the next N steps' keys, whose
#: order is knowable from the loader's seeded shuffle, instead of all of
#: them) is the answer this threshold exists to flag the need for.
#:
#: Deliberately not a Port. It is a diagnostic, not a control: passing a
#: larger budget does not make warming a million captions a good idea, it
#: just moves the machine's failure later. Raise it deliberately in this
#: file once a windowed prewarm exists and this stops being the guard.
PREWARM_HOST_RAM_BUDGET_BYTES = 8 * 1024 ** 3

#: Bytes one prompt entry actually occupies, measured on the B580 by
#: summing the stored tensors (a (1, 77, 2048) float32 context plus ~0.3 KB
#: pooled): 621 KB. Used to turn the RAM budget above into an entry count.
#: Deliberately a measured constant rather than a guess from tensor shapes,
#: because the whole point of the budget is to be right about a number that
#: is otherwise invisible until the machine runs out.
PREWARM_BYTES_PER_PROMPT_ENTRY = 621 * 1024

#: How many prompt entries to warm at most. **5,000**, the project's
#: starting figure (2026-10-04), which is also about what the host-RAM
#: budget above allows on its own terms: 5,000 x 621 KB = 3.0 GiB of the
#: 8 GiB, so the count is the one that binds and the byte budget is the
#: cross-check that it is sane.
#:
#: Both terms of warming are measured, and at this size neither is a
#: problem: 5,000 prompts is 12.7 s batched (55 s in fp32, see
#: `docs/design/09-prioritized-backlog.md`) against roughly an hour of
#: training to *use* those 5,000 prompts at batch 2. So this limit is not
#: about time. It is about the resident 3.0 GiB, and about what a miss
#: costs if the dataset has more prompts than this: ~790 ms measured.
#:
#: Raise it deliberately. It is not a Port because passing a bigger number
#: here does not make a larger cache a good idea on a machine with finite
#: RAM; it just decides where the ceiling is rather than what happens at it.
PREWARM_MAX_PROMPT_ENTRIES = 5_000

#: Floor on the stop threshold: consecutive batches with no new prompt before
#: discovery gives up, when only a few prompts have been seen.
#:
#: The discovery pass costs **1.24 ms per sample** on the B580 (measured: 201
#: samples in 0.25 s), because it reads every latent off disk to read its
#: shape. Against that, warming a prompt costs **30 ms once**. So on a dataset
#: with few distinct prompts the pass is almost all cost and no benefit: at
#: 1M samples it is **21 minutes to warm one prompt**, and all three of this
#: project's real datasets have exactly one.
STOP_AFTER_NO_NEW_PROMPTS = 64

#: Multiple of the prompts seen so far, added to that floor.
#:
#: A fixed threshold does not work, and the way it fails is worth recording
#: because it looks like a bug in the other direction. With 200 prompts
#: shuffled into 1,000 batches, a threshold of 64 *does* trip -- late in the
#: pass, once nearly everything has been seen, the gap between consecutive new
#: prompts grows like a coupon collector's, and runs of 64 stale batches are
#: ordinary by then. It trips having missed nothing that was left to miss, so
#: it is not wrong, only premature about stopping.
#:
#: Scaling the threshold by what has been found is the fix: the question is
#: not "has anything new appeared lately" but "have I seen enough that nothing
#: new appearing lately means there is nothing left". After k prompts, a new
#: one appears roughly every (total - k)/k batches, so the stale run has to
#: grow with k. At 8x, one prompt trips at 64 batches and 200 prompts would
#: need 1,600 -- longer than the dataset, so the pass completes.
#:
#: The bound still converges: once every prompt *has* been seen, no threshold
#: would help, and stopping is the correct answer, because there is genuinely
#: nothing further to find.
STALE_BATCHES_PER_PROMPT_SEEN = 8


def prompt_capacity() -> int:
    """Prompt entries to warm at most: the count cap, floored at 1.

    Deliberately *not* derived from `PREWARM_HOST_RAM_BUDGET_BYTES` alone
    any more. At 8 GiB / 621 KB that would be 13,508 entries, and the
    project's figure is 5,000 -- a count is the limit someone chose rather
    than one an arithmetic fallback produced, and the two only agreed to
    within a factor of 2.7 by luck. The byte budget stays as the check
    that the chosen count is affordable.
    """
    return max(1, PREWARM_MAX_PROMPT_ENTRIES)


class StopReason(str, Enum):
    """Why a discovery pass ended. Reported, never swallowed."""
    COMPLETED = "completed"
    NO_NEW_PROMPTS = "no_new_prompts"
    MAX_BATCHES = "max_batches"


@dataclass(frozen=True, slots=True)
class Discovery:
    """What a discovery pass found, and how it ended.

    Not a bare set, because "found every key" and "stopped looking and
    missed some" are the same value with completely different consequences,
    and a bare set cannot tell a caller which it has. The second is not wrong
    -- an unwarmed key is a cache miss, and a miss answers correctly -- but
    it is slower, it puts CLIP back on the card, and on a big dataset it is
    the difference between 0.08 s and 21 minutes. So it is visible.
    """

    keys: frozenset
    batches_seen: int
    prompts_found: int
    reason: StopReason

    @property
    def truncated(self) -> bool:
        return self.reason is not StopReason.COMPLETED


def discover_dataset_keys(dataset: TrainingBatchSource,
                          max_batches: int | None = None,
                          stop_after_no_new_prompts: int | None = STOP_AFTER_NO_NEW_PROMPTS
                          ) -> Discovery:
    """One real pass over `dataset` collecting every (prompt,
    batch_size, height, width) key training will request -- derived
    exactly the way nodes/train/step_pipeline.py's
    EncodeConditioningPhase derives them at request time (x_t.shape
    gives batch_size and, *8 for the VAE downsample factor,
    height/width).

    `dataset` must be finite per iteration (one pass = one epoch;
    ManagedDatasetLoader-backed sources are exactly that -- training's
    own FetchBatchPhase starts a fresh `iter()` for the next one).
    Order doesn't matter for the *set*: it's over the whole source, so a
    per-epoch shuffle cannot hide a key -- which is what makes the stop
    condition below safe, and also what bounds where it is not.

    **Two ways to stop early, both benign, both reported.**

    `max_batches` bounds the assumption that the source ends at all, which
    matters now that prewarm is on by default: a source that never ends would
    otherwise hang the trainer at startup with no output and no error, the
    worst way for a default to fail.

    `stop_after_no_new_prompts` bounds the *cost*, which is the larger
    problem. This pass reads every latent off disk to read its shape --
    1.24 ms per sample on the B580 -- to collect keys that cost 30 ms each to
    warm, so a dataset whose prompts are few pays almost entirely for
    nothing. Stopping after N consecutive batches that introduce no new
    prompt turns 1M samples into N. See `STOP_AFTER_NO_NEW_PROMPTS`.

    Truncation is never wrong. A key past the stop point is a cache miss, and
    a miss self-loads the encoder and returns the right answer -- so the
    failure mode is "slower, and CLIP back on the card for the keys nobody
    warmed". What it would cost on a 1M-sample dataset is the difference
    between warming one prompt in 0.08 s and in 21 minutes, which is why the
    bound exists at all.
    """
    unique_keys: set = set()
    prompt_keys: set = set()
    stale_batches = 0
    seen_batches = 0
    reason = StopReason.COMPLETED

    for batch in dataset:
        seen_batches += 1
        if max_batches is not None and seen_batches > max_batches:
            reason = StopReason.MAX_BATCHES
            break
        height = batch["x_t"].shape[2] * 8
        width = batch["x_t"].shape[3] * 8
        prompt_key = (batch["prompt"], batch["x_t"].shape[0])
        key = (prompt_key[0], prompt_key[1], height, width)
        # Counted on the **whole key**, resolution included -- and that is a
        # correction, not the obvious choice. A new resolution is cheap to
        # warm (1.6 ms against a prompt's 30 ms), which is what made it look
        # safe to ignore. Measured on the real data, it is not: `non-square`
        # has 44 resolutions spread over 121 batches, so stopping on prompts
        # alone at batch 65 missed 43 of them -- and a *missed* key costs
        # ~790 ms, because the miss self-loads CLIP. Cheap to warm and cheap
        # to miss are different things, and only the first one is true.
        if key in unique_keys:
            stale_batches += 1
        else:
            unique_keys.add(key)
            stale_batches = 0
        prompt_keys.add(prompt_key)
        threshold = None
        if stop_after_no_new_prompts is not None:
            threshold = max(stop_after_no_new_prompts,
                            STALE_BATCHES_PER_PROMPT_SEEN * len(unique_keys))
        if threshold is not None and stale_batches >= threshold:
            reason = StopReason.NO_NEW_PROMPTS
            break

    if reason is StopReason.MAX_BATCHES:
        logger.warning(
            "discover_dataset_keys: stopped at the %d-batch cap without the "
            "source ending; warming %d key(s) found so far. Any key past that "
            "point is a cache miss -- correct but slow, and it re-loads CLIP. "
            "Wire a source that is finite per iteration (ManagedDatasetLoader-"
            "backed ones are) to get the full prewarm.",
            max_batches, len(unique_keys),
        )
    elif reason is StopReason.NO_NEW_PROMPTS:
        logger.info(
            "discover_dataset_keys: stopped after %d batches -- %d in a row "
            "introduced no new prompt -- having found %d distinct prompt(s) "
            "and %d key(s). Any key past that point is a cache miss: correct, "
            "but it re-loads CLIP. The threshold scales with the prompts seen "
            "so far (%d x), so a shuffled source does not trip it early; if "
            "this fired while prompts were still appearing, raise "
            "STALE_BATCHES_PER_PROMPT_SEEN.",
            seen_batches, stale_batches, len(prompt_keys), len(unique_keys),
            STALE_BATCHES_PER_PROMPT_SEEN,
        )
    return Discovery(keys=frozenset(unique_keys), batches_seen=seen_batches,
                     prompts_found=len(prompt_keys), reason=reason)


def warm_and_unload(cached: CachingTextEncoder, discovery: Discovery) -> int:
    """Encode as many discovered keys into `cached` as its budget allows,
    then `cached.unload()` the underlying encoder entirely. Returns the
    number of keys the dataset had.

    Shared by this module's own node (main route) and
    `ManagedLoRATrainerNode`'s `prewarm_text_encoder` Port (Resources
    Controller route). Keys outside `keys` later (dataset changed between
    warm and training) miss this cache -- correct but slow, see this
    module's docstring's degradation note.

    **Capacity comes from the host-RAM budget, not from `len(keys)`.** That
    inversion is the whole change: sizing the cache to the dataset made host
    RAM a function of dataset size, which is the coupling that stops prewarm
    scaling (a 1M-caption dataset wanted 606 GB). It also had a quieter
    failure -- with capacity below the distinct-prompt count, the warm pass's
    tail encodes, inserts, and is LRU-evicted on the very next insert, so it
    warms *nothing* for those prompts and they every one miss later. Capping
    the cache therefore has to come with capping what gets warmed, which is
    what the prompt filter below does. Nothing else does this for it:
    `CachingTextEncoder` has no way to know a warm pass is coming.

    Resolution keys are ~0.3 KB against a prompt entry's 621 KB, so they are
    deliberately not filtered on the same budget -- `non-square` alone has 43
    of them against 1 prompt, at 1.6 ms each. All of them are warmed.

    **What warming costs, reported, because the cost decides this.** Both
    terms are linear in distinct prompts and neither is visible from outside:
    time (30.6 ms each on XPU, 1,505 ms on CPU -- measured on the B580) and
    host RAM (621 KB each, summed exactly by `cache_bytes()`).
    """
    keys = discovery.keys
    prompt_keys = sorted({(p, bs) for p, bs, _, _ in keys})
    resolution_keys = sorted({(h, w, bs) for _, bs, h, w in keys})
    capacity = prompt_capacity()
    warmable = prompt_keys[:capacity]
    skipped = len(prompt_keys) - len(warmable)
    # Prompt half first, so a truncated warm still fills the cache rather
    # than spending its budget on resolution keys. (Both halves are cached
    # separately, so warming the prompt half needs no resolution at all.)
    started = time.monotonic()
    # The prompt half first, so a truncated warm still fills the cache rather
    # than spending its budget on resolution keys. (Both halves are cached
    # separately, so the prompt half needs no resolution at all.)
    #
    # The first prompt goes through `encode` because it pays the one-time
    # device setup, and it is what makes the resolution loop below legal --
    # the cache's `encode` needs both halves' arguments even when only one
    # of them misses. The rest go through `warm_prompts`, which batches.
    cached.encode(warmable[0][0], warmable[0][1],
                  resolution_keys[0][0], resolution_keys[0][1])
    setup = time.monotonic() - started
    started = time.monotonic()
    # Grouped by batch_size because that is part of the cache key, so one
    # bulk call has to be homogeneous to stay a bulk call. In practice every
    # key shares a batch_size -- they come from one dataset at one batch_size
    # -- so this is one group, and the grouping exists so that a caller
    # passing mixed keys gets correct keys rather than a fast wrong answer.
    by_batch: dict[int, list[str]] = {}
    for prompt, batch_size in warmable[1:]:
        by_batch.setdefault(batch_size, []).append(prompt)
    for batch_size, group in by_batch.items():
        cached.warm_prompts(group, batch_size)
    prompt_time = time.monotonic() - started
    started = time.monotonic()
    for height, width, batch_size in resolution_keys:
        cached.encode(warmable[0][0], warmable[0][1], height, width)
    resolution_time = time.monotonic() - started
    cached_bytes = getattr(cached, "cache_bytes", lambda: 0)()
    cached.unload()

    others = len(warmable) - 1
    marginal = 1000 * prompt_time / others if others > 0 else 0.0
    # print, not logging: these are the numbers an operator is meant to see
    # and decide by, and this project's convention for that is stdout (the
    # loader's own one-time warning, and EncodeConditioningPhase's residency
    # lines, both print). A logging call at INFO is invisible unless
    # something configured logging, which nothing here does -- a
    # measurement nobody can see is not a measurement.
    scanned = (f"found in {discovery.batches_seen} batch(es)"
               if discovery.truncated else "found by reading the whole dataset")
    print(f"  [prewarm] {len(keys)} key(s) {scanned}: {len(prompt_keys)} "
          f"distinct prompt(s), {len(resolution_keys)} resolution. Warmed "
          f"{len(warmable)} prompt(s) in {setup + prompt_time:.2f}s "
          f"({setup:.2f}s one-time setup"
          + (f", {marginal:.1f} ms per further prompt" if others > 0 else "")
          + f") and {len(resolution_keys)} resolution key(s) in "
          f"{resolution_time:.2f}s, holding {cached_bytes / 2 ** 20:.1f} MB of "
          f"host RAM. CLIP is off the card for the rest of the run.")
    if others > 0 and not getattr(cached, "batching_available", lambda: True)():
        # Said out loud, because the alternative is a warm pass that is 3x
        # slower than it could be with no visible reason why.
        print(
            "  [prewarm] NOTE: prompt encoding was one-at-a-time, not "
            "batched -- the encoder is not in float32, and batching in "
            "float16 would disagree with the per-prompt cache-miss path by "
            "19% (nodes/model/clip_encoder.py's encode_prompts has the "
            "measurement). Correct and slower; load CLIP in float32 to get "
            "the batched path.",
            flush=True,
        )
    if discovery.truncated:
        print(
            f"  [prewarm] NOTE: discovery stopped early ({discovery.reason.value}) "
            f"after {discovery.batches_seen} batch(es), having found "
            f"{discovery.prompts_found} distinct prompt(s). Any key past that "
            f"point is a cache miss: correct, but it re-loads CLIP and costs "
            f"~790 MB's worth of it the first time. The threshold scales with "
            f"prompts found, so shuffling is safe; a source whose prompts are "
            f"clustered *and* whose tail is long enough would trip it.",
            flush=True,
        )
    if skipped:
        print(
            f"  [prewarm] WARNING: {skipped} of {len(prompt_keys)} distinct "
            f"prompt(s) did NOT fit the {PREWARM_HOST_RAM_BUDGET_BYTES / 1024 ** 3:.0f} GB "
            f"host-RAM budget ({capacity:,} entries at "
            f"{PREWARM_BYTES_PER_PROMPT_ENTRY // 1024} KB each) and were not "
            f"warmed. They will be cache misses on first use: correct, and it "
            f"re-loads CLIP at ~790 ms each (364 load + 30 encode + 396 evict, "
            f"measured) rather than the {marginal:.0f} ms a warm one costs. "
            f"Raise the budget deliberately if this dataset is worth it.",
            flush=True,
        )
    return len(keys)


class PrewarmedTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "encoder": Port(name="encoder", type=TextEncoder, required=True,
                         doc="The real encoder to warm and then unload, e.g. an "
                             "SDXLTextEncoderNode's output. Main route only -- the "
                             "Resources Controller route has no encoder graph port; "
                             "use ManagedLoRATrainerNode's prewarm_text_encoder Port "
                             "there (see this module's own docstring)."),
        "dataset": Port(
            name="dataset", type=TrainingBatchSource, required=True,
            doc="One full pass is taken over this to discover every (prompt, batch_size, "
                "height, width) combination training will request -- must be the same "
                "dataset (with the same wiring, e.g. through a PrefetchingBatchSourceNode "
                "if one's used) that actually gets wired into the trainer, not a subset of it.",
        ),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        encoder: TextEncoder = inputs["encoder"]
        dataset: TrainingBatchSource = inputs["dataset"]

        discovery = discover_dataset_keys(
            dataset, max_batches=MAX_DISCOVERY_BATCHES)
        cached = CachingTextEncoder(
            encoder, max_entries=max(len(discovery.keys), 1))
        warm_and_unload(cached, discovery)

        result = {"encoder": cached}
        self.validate_outputs(result)
        return result
