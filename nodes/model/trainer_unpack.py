"""TrainerResourcesUnpackNode: exposes a LoRATrainingConfigNode's
bundled `trainer` (LoRATrainingSkeleton) output as the plain
model/text_encoder ports every existing consumer already understands --
the piece that actually unblocks the Resources Controller route into
TrainerNode (docs/design/resources-controller/06-phase-6-lora-training-config.md's
"Decided, not yet built" TrainerNode-integration item; see that file's
own updated status for the full reasoning this corrects).

Why this exists instead of widening TrainerNode's own ports to accept
`trainer` directly, or replacing its model/text_encoder ports with one
bundled `trainer` port (the shape that file originally sketched):
server/graph_executor.py's `_is_compatible()` does a real issubclass()
check on every wire (docs/design/resources-controller/
01-context-and-ground-truth.md's own "every edge in the graph is
type-checked before any node runs" -- checked directly, not assumed).
`LoRATrainingSkeleton` is not a `TrainableModel` or a `TextEncoder` --
it *has* one of each, as `.unet`/`.clip` (composition, not inheritance,
per this project's own `SDXLArchitecture`/`LoRATrainingSkeleton`
design) -- so nothing produced by `ResourcesControllerNode` ->
`LoRATrainingConfigNode`'s route could wire directly into
`ModelParametersNode.model`, `CachingTextEncoderNode.encoder`, or
`TrainerNode.model`/`.text_encoder` without something bridging the two
type worlds. Replacing TrainerNode's own ports instead would have
broken the existing, already-working main route (ComfyUNetLoRANode's
`model` output and SDXLTextEncoderNode's `text_encoder` output are
never wrapped in a LoRATrainingSkeleton at all) -- not an acceptable
trade for a route explicitly meant to run *alongside* the main one for
comparison, not replace it before it's even been shown to work better.

Kept deliberately thin (mirrors nodes/model/parameters.py's own
ModelParametersNode in spirit and size) -- .unet/.clip are already
real, fully-built objects by the time `trainer` exists
(LoRATrainingConfigNode.build() already did the actual work); there's
nothing left to do here but expose the two attributes this project's
existing nodes already know how to consume. Effect: the entire rest of
the graph -- ModelParametersNode, CachingTextEncoderNode,
PrewarmedTextEncoderNode, SupervisedLoRATrainerNode,
BudgetedLoRATrainerNode (nodes/train/budgeted.py),
LoRACheckpointSaverNode -- is reusable on the new route completely
unmodified; none of it is duplicated or specialized for this route.
`vae_sd` and `lora` (the continue-from-LoRA reference, already loaded
into the adapter by the time `trainer` exists) are deliberately not
exposed here -- nothing under nodes/train/ needs either today (checked
directly: no VAE-decode/preview node exists yet under nodes/), so
there's nothing real to wire them into. Add them here, not on a second
unpack node, if and when that changes.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Node, Port
from .handle import TrainableModel
from .lora_training_resources import LoRATrainingSkeleton
from .text_encoder import TextEncoder


class TrainerResourcesUnpackNode(Node):

    INPUTS: ClassVar[dict[str, Port]] = {
        "trainer": Port(
            name="trainer", type=LoRATrainingSkeleton, required=True,
            doc="A LoRATrainingConfigNode's own `trainer` output -- real, LoRA-injected "
                "unet/clip, ready to train.",
        ),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(
            name="model", type=TrainableModel, required=True,
            doc="trainer.unet -- feed into ModelParametersNode (for an optimizer node's "
                "params input) and into a trainer node's own model input.",
        ),
        "text_encoder": Port(
            name="text_encoder", type=TextEncoder, required=True,
            doc="trainer.clip -- feed into a trainer node's text_encoder input directly, "
                "or through CachingTextEncoderNode first to make it offload-eligible under "
                "a VRAM budget.",
        ),
    }

    def build(self, **inputs) -> dict[str, object]:
        self.validate_inputs(inputs)
        trainer: LoRATrainingSkeleton = inputs["trainer"]
        result = {"model": trainer.unet, "text_encoder": trainer.clip}
        self.validate_outputs(result)
        return result
