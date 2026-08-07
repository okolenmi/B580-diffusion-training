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
in, not re-solving caching.

If something outside the discovered set is ever requested anyway (the
dataset changes between this node running and training starting, or any
other mismatch), this degrades, not breaks: the underlying encoder is
only moved to CPU by unload(), not destroyed, so a genuine cache miss
still returns a correct answer -- just recomputed on CPU instead of the
accelerator, and it wouldn't be there in the first place. Not something
to be usually relied on, but not a hard failure mode either.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Port
from ..dataset.handle import TrainingBatchSource
from .text_encoder import TextEncoder, TextEncoderNode
from .text_encoder_cache import CachingTextEncoder


class PrewarmedTextEncoderNode(TextEncoderNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        "encoder": Port(name="encoder", type=TextEncoder, required=True,
                         doc="The real encoder to warm and then unload, e.g. an "
                             "SDXLTextEncoderNode's output."),
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

        unique_keys = set()
        for batch in dataset:
            height = batch["x_t"].shape[2] * 8
            width = batch["x_t"].shape[3] * 8
            unique_keys.add((batch["prompt"], batch["x_t"].shape[0], height, width))

        cached = CachingTextEncoder(encoder, max_entries=max(len(unique_keys), 1))
        for prompt, batch_size, height, width in unique_keys:
            cached.encode(prompt, batch_size, height, width)
        cached.unload()

        result = {"encoder": cached}
        self.validate_outputs(result)
        return result
