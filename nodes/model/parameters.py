"""ModelParametersNode: extracts a ParameterList from a TrainableModel.

The connecting piece between the model domain and OptimizerNode.params --
without this, there was nothing that actually produced a ParameterList,
so the only thing wireable into an optimizer's params input was itself,
via a widget, which doesn't make sense for a handle-typed port. Also the
answer to "what do I connect to params": this node, not the model output
directly.
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Node, Port
from .handle import ParameterList, TrainableModel


class ModelParametersNode(Node):

    INPUTS: ClassVar[dict[str, Port]] = {
        "model": Port(name="model", type=TrainableModel, required=True,
                       doc="The model to pull trainable parameters from."),
    }
    OUTPUTS: ClassVar[dict[str, Port]] = {
        "params": Port(name="params", type=ParameterList, required=True,
                        doc="Feed this into an optimizer node's params input."),
    }

    def build(self, **inputs) -> dict[str, ParameterList]:
        self.validate_inputs(inputs)
        model: TrainableModel = inputs["model"]
        result = {"params": ParameterList(model.trainable_parameters())}
        self.validate_outputs(result)
        return result
