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

#: How many prompt entries that budget buys -- ~13,200 at 8 GiB.
PREWARM_MAX_PROMPT_ENTRIES = (
    PREWARM_HOST_RAM_BUDGET_BYTES // PREWARM_BYTES_PER_PROMPT_ENTRY
)


def prompt_capacity() -> int:
    """Prompt entries the host-RAM budget allows, floor 1."""
    return max(1, PREWARM_MAX_PROMPT_ENTRIES)


def discover_dataset_keys(dataset: TrainingBatchSource,
                          max_batches: int | None = None) -> set:
    """One real pass over `dataset` collecting every (prompt,
    batch_size, height, width) key training will request -- derived
    exactly the way nodes/train/step_pipeline.py's
    EncodeConditioningPhase derives them at request time (x_t.shape
    gives batch_size and, *8 for the VAE downsample factor,
    height/width).

    `dataset` must be finite per iteration (one pass = one epoch;
    ManagedDatasetLoader-backed sources are exactly that -- training's
    own FetchBatchPhase starts a fresh `iter()` for the next one).
    Order doesn't matter: it's a set over the whole source, so a
    per-epoch shuffle can't hide a key.

    **`max_batches` bounds that assumption instead of trusting it**, which
    matters now that prewarm is on by default: a source that does *not*
    end would otherwise hang the trainer at startup with no output and no
    error, which is the worst way for a default to fail. On reaching the
    bound it stops and returns what it found, with a warning naming what
    that costs. Truncation is benign rather than wrong: the keys missed
    are cache misses, and a miss self-loads the encoder and returns the
    right answer. So the failure mode is "slower and CLIP back on the
    card for the keys nobody warmed", never a wrong training step.
    """
    unique_keys = set()
    for seen, batch in enumerate(dataset, start=1):
        if max_batches is not None and seen > max_batches:
            logger.warning(
                "discover_dataset_keys: stopped after %d batches without the "
                "source ending; warming %d key(s) found so far. Any key past "
                "that point will be a cache miss -- correct but slow, and it "
                "re-loads CLIP. Wire a source that is finite per iteration "
                "(ManagedDatasetLoader-backed ones are) to get the full "
                "prewarm.", max_batches, len(unique_keys),
            )
            break
        height = batch["x_t"].shape[2] * 8
        width = batch["x_t"].shape[3] * 8
        unique_keys.add((batch["prompt"], batch["x_t"].shape[0], height, width))
    return unique_keys


def warm_and_unload(cached: CachingTextEncoder, keys: set) -> int:
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
    prompt_keys = sorted({(p, bs) for p, bs, _, _ in keys})
    resolution_keys = sorted({(h, w, bs) for _, bs, h, w in keys})
    capacity = prompt_capacity()
    warmable = prompt_keys[:capacity]
    skipped = len(prompt_keys) - len(warmable)
    # Prompt half first, so a truncated warm still fills the cache rather
    # than spending its budget on resolution keys. (Both halves are cached
    # separately, so warming the prompt half needs no resolution at all.)
    started = time.monotonic()
    cached.encode(warmable[0][0], warmable[0][1],
                  resolution_keys[0][0], resolution_keys[0][1])
    setup = time.monotonic() - started
    started = time.monotonic()
    for prompt, batch_size in warmable[1:]:
        cached.encode(prompt, batch_size,
                      resolution_keys[0][0], resolution_keys[0][1])
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
    print(f"  [prewarm] {len(keys)} key(s) in the dataset: {len(prompt_keys)} "
          f"distinct prompt(s), {len(resolution_keys)} resolution. Warmed "
          f"{len(warmable)} prompt(s) in {setup + prompt_time:.2f}s "
          f"({setup:.2f}s one-time setup"
          + (f", {marginal:.1f} ms per further prompt" if others > 0 else "")
          + f") and {len(resolution_keys)} resolution key(s) in "
          f"{resolution_time:.2f}s, holding {cached_bytes / 2 ** 20:.1f} MB of "
          f"host RAM. CLIP is off the card for the rest of the run.")
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

        keys = discover_dataset_keys(dataset, max_batches=MAX_DISCOVERY_BATCHES)
        cached = CachingTextEncoder(encoder, max_entries=max(len(keys), 1))
        warm_and_unload(cached, keys)

        result = {"encoder": cached}
        self.validate_outputs(result)
        return result
