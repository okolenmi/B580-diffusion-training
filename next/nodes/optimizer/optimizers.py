"""One node per update rule. Each builds an optimizer over the model's adapter parameter groups."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from ...engine.handles import TrainableDenoiser
from ...engine.optim.algorithms import CAME, AdamW, Adafactor, Algorithm
from ...engine.optim.handle import FusedOptimizer, Optimizer
from ...engine.optim.state import STORES, make_store
from ...graph import Group, Node, Port


class OptimizerNode(Node):
    CATEGORY = "Optimizer"
    GROUPS = (Group("rule", "Update rule"), Group("execution", "Execution", collapsed=True, advanced=True))
    COMMON = {
        "model": Port("model", TrainableDenoiser),
        "lr": Port("lr", float, required=False, default=1e-4, minimum=0.0, group="rule",
                   doc="Base learning rate. Parameter groups may multiply it (see the adapter's groups)."),
        "grad_clip": Port("grad_clip", float, required=False, default=0.0, minimum=0.0, group="rule",
                          doc="Largest total gradient norm; 0 turns clipping off (not available when fused)."),
        "fused": Port("fused", bool, required=False, default=False, group="execution",
                      doc="Update each parameter as soon as its gradient is ready, so gradients never pile up."),
        "group_size": Port("group_size", int, required=False, default=0, minimum=0, group="execution",
                           doc="Parameters updated together: 0 all same-shaped ones (fastest), 1 one at a time (least memory)."),
        "foreach": Port("foreach", bool, required=False, default=True, group="execution", visible_when=("fused", False)),
        "state_precision": Port("state_precision", str, required=False, default="float32", choices=tuple(STORES), group="execution",
                                doc="int8_blockwise keeps the optimizer state at about a quarter of the memory, with a small precision loss."),
    }
    OUTPUTS = {"optimizer": Port("optimizer", Optimizer)}

    @abstractmethod
    def algorithm(self, inputs: dict) -> Algorithm: ...

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        model: TrainableDenoiser = inputs["model"]
        fused = bool(inputs.get("fused"))
        cls = FusedOptimizer if fused else Optimizer
        kwargs = {"group_size": inputs.get("group_size") or 0, "store": make_store(inputs.get("state_precision") or "float32"),
                  "foreach": False if fused else inputs.get("foreach", True), "grad_clip": inputs.get("grad_clip") or None}
        return {"optimizer": cls(self.algorithm(inputs), model.param_groups(), inputs.get("lr", 1e-4), **kwargs)}


def _pair(value, default):
    return tuple(value) if value is not None else default


class AdamWNode(OptimizerNode):
    DISPLAY_NAME = "AdamW"
    INPUTS = {**OptimizerNode.COMMON,
              "beta1": Port("beta1", float, required=False, default=0.9, group="rule"),
              "beta2": Port("beta2", float, required=False, default=0.999, group="rule"),
              "eps": Port("eps", float, required=False, default=1e-8, group="rule"),
              "weight_decay": Port("weight_decay", float, required=False, default=0.01, group="rule")}

    def algorithm(self, inputs):
        return AdamW((inputs.get("beta1", 0.9), inputs.get("beta2", 0.999)), inputs.get("eps", 1e-8), inputs.get("weight_decay", 0.01))


class AdafactorNode(OptimizerNode):
    DISPLAY_NAME = "Adafactor"
    INPUTS = {**OptimizerNode.COMMON,
              "beta1": Port("beta1", float, required=False, default=0.0, minimum=0.0, maximum=0.999, group="rule",
                            doc="First-moment decay; 0 keeps no first moment (least memory)."),
              "clip_threshold": Port("clip_threshold", float, required=False, default=1.0, group="rule"),
              "scale_parameter": Port("scale_parameter", bool, required=False, default=False, group="rule",
                                      doc="Scale each step by the parameter's own size."),
              "weight_decay": Port("weight_decay", float, required=False, default=0.0, group="rule")}

    def algorithm(self, inputs):
        return Adafactor(beta1=inputs.get("beta1") or None, clip_threshold=inputs.get("clip_threshold", 1.0),
                         scale_parameter=bool(inputs.get("scale_parameter")), weight_decay=inputs.get("weight_decay", 0.0))


class CAMENode(OptimizerNode):
    DISPLAY_NAME = "CAME"
    INPUTS = {**OptimizerNode.COMMON,
              "beta1": Port("beta1", float, required=False, default=0.9, group="rule"),
              "beta2": Port("beta2", float, required=False, default=0.999, group="rule"),
              "beta3": Port("beta3", float, required=False, default=0.9999, group="rule"),
              "clip_threshold": Port("clip_threshold", float, required=False, default=1.0, group="rule"),
              "weight_decay": Port("weight_decay", float, required=False, default=0.0, group="rule")}

    def algorithm(self, inputs):
        return CAME(clip_threshold=inputs.get("clip_threshold", 1.0),
                    betas=(inputs.get("beta1", 0.9), inputs.get("beta2", 0.999), inputs.get("beta3", 0.9999)),
                    weight_decay=inputs.get("weight_decay", 0.0))
