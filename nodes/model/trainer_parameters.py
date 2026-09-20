"""TrainerParametersNode: extracts a ParameterList from a
LoRATrainingSkeleton's own `.unet` -- the Resources Controller route's
own counterpart to ModelParametersNode (nodes/model/parameters.py),
which only accepts a plain TrainableModel and can't take a
LoRATrainingSkeleton directly (it isn't one -- it *has* one, as
`.unet`).

A second, independent node rather than widening ModelParametersNode's
own `model` port: this route's own trainer
(ManagedLoRATrainerNode, nodes/train/managed.py) takes `trainer`
directly rather than the plain model/text_encoder ports the main
route's SupervisedLoRATrainerNode does, so there's no shared
`TrainableModel`-typed port left in this route for ModelParametersNode
to wire from at all -- this is that route's own, equally thin,
connecting piece between LoRATrainingConfigNode's `trainer` output and
an optimizer node's `params` input.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Node, Port
from .handle import ParameterList
from .lora_training_resources import LoRATrainingSkeleton


class TrainerParametersNode(Node):

    INPUTS: ClassVar[dict[str, Port]] = {
        "trainer": Port(
            name="trainer", type=LoRATrainingSkeleton, required=True,
            doc="A LoRATrainingConfigNode's own `trainer` output -- pulls trainable "
                "parameters from its .unet.",
        ),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "params": Port(name="params", type=ParameterList, required=True,
                        doc="Feed this into an optimizer node's params input."),
    }

    def build(self, **inputs) -> dict[str, ParameterList]:
        self.validate_inputs(inputs)
        trainer: LoRATrainingSkeleton = inputs["trainer"]
        result = {"params": ParameterList(trainer.unet.trainable_parameters())}
        self.validate_outputs(result)
        return result
