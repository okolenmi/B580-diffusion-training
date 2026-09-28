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

from typing import ClassVar

from ..core import Port
from ..dataset.handle import TrainingBatchSource
from .text_encoder import TextEncoder, TextEncoderNode
from .text_encoder_cache import CachingTextEncoder


def discover_dataset_keys(dataset: TrainingBatchSource) -> set:
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
    """
    unique_keys = set()
    for batch in dataset:
        height = batch["x_t"].shape[2] * 8
        width = batch["x_t"].shape[3] * 8
        unique_keys.add((batch["prompt"], batch["x_t"].shape[0], height, width))
    return unique_keys


def warm_and_unload(cached: CachingTextEncoder, keys: set) -> int:
    """Encode every discovered key into `cached`, then `cached.unload()`
    the underlying encoder entirely. Returns the number of keys warmed.

    Shared by this module's own node (main route) and
    ManagedLoRATrainerNode's `prewarm_text_encoder` Port (Resources
    Controller route) -- both callers wrap in CachingTextEncoder first
    (this node sizing max_entries to len(keys) so the warm pass can't
    evict itself; a caller-supplied cache keeps its own capacity), this
    only does the warm + unload. Keys outside `keys` later (dataset
    changed between warm and training) miss this cache -- correct but
    slow, see this module's docstring's degradation note.
    """
    for prompt, batch_size, height, width in keys:
        cached.encode(prompt, batch_size, height, width)
    cached.unload()
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
                "dataset (with the same wiring, e.g. through a RenoiseBatchSourceNode if "
                "one's used) that actually gets wired into the trainer, not a subset of it.",
        ),
    }

    def build(self, **inputs) -> dict[str, TextEncoder]:
        self.validate_inputs(inputs)
        encoder: TextEncoder = inputs["encoder"]
        dataset: TrainingBatchSource = inputs["dataset"]

        keys = discover_dataset_keys(dataset)
        cached = CachingTextEncoder(encoder, max_entries=max(len(keys), 1))
        warm_and_unload(cached, keys)

        result = {"encoder": cached}
        self.validate_outputs(result)
        return result
