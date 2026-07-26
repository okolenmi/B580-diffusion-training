"""ComfyUNetLoRANode: builds comfy's SDXL UNet from raw weights, injects LoRA.

Adapter only -- core.unet_wrapper.ComfyUNetWrapper and core.lora already do
the real work (UNet construction, LoRA layer injection); this wraps them
behind the TrainableModel/LoRAInjectorNode contracts.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..core import Port
from .handle import ModelWeights, TrainableModel
from .node import LoRAInjectorNode


class ComfyUNetTrainableModel(TrainableModel):

    def __init__(self, wrapper):
        self._wrapper = wrapper

    def forward(self, x_t, timestep, context, y):
        return self._wrapper.forward(x_t, timestep, context, y)

    def trainable_parameters(self) -> list:
        return self._wrapper.lora_parameters()

    def train(self) -> "ComfyUNetTrainableModel":
        self._wrapper.train()
        return self

    def eval(self) -> "ComfyUNetTrainableModel":
        self._wrapper.eval()
        return self

    def to(self, device=None, **kwargs) -> "ComfyUNetTrainableModel":
        self._wrapper.to(device=device, **kwargs)
        return self

    def state_dict(self) -> dict:
        return self._wrapper.get_lora_weights()

    @property
    def raw(self):
        """Escape hatch to the wrapped ComfyUNetWrapper, for callers (e.g.
        LoRACheckpointSaverNode) that need the full legacy object."""
        return self._wrapper


class ComfyUNetLoRANode(LoRAInjectorNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **LoRAInjectorNode.COMMON_INPUTS,
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "dtype": Port(name="dtype", type=Any, required=False, default=None,
                       doc="torch dtype; None resolves to torch.bfloat16."),
        "rank": Port(name="rank", type=int, required=False, default=64),
        "alpha": Port(name="alpha", type=float, required=False, default=1.0),
        "dropout": Port(name="dropout", type=float, required=False, default=0.0),
        "target_modules": Port(name="target_modules", type=Any, required=False, default=None),
        "use_checkpoint": Port(
            name="use_checkpoint", type=bool, required=False, default=False,
            doc="Gradient checkpointing. Defaults to False for LoRA training -- ComfyUI's "
                "own checkpoint() (comfy/ldm/modules/diffusionmodules/util.py) passes an "
                "entire block's self.parameters() into torch.autograd.grad()'s inputs= list "
                "on every backward, not filtered to the ones that actually require grad. "
                "With a frozen base + LoRA, essentially every block has at least one frozen "
                "parameter (a norm weight, a bias, anything LoRA didn't target), and that "
                "call raises 'One of the differentiated Tensors does not require grad' the "
                "moment it hits one. True trades VRAM for that crash; only turn it on if "
                "every parameter in every checkpointed block is itself LoRA-adapted, which "
                "target_modules would have to be built specifically to guarantee.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)
        import torch
        from core.lora import LoRAConfig
        from core.unet_wrapper import ComfyUNetWrapper

        weights: ModelWeights = inputs["weights"]
        lora_config = LoRAConfig(
            rank=inputs.get("rank", self.INPUTS["rank"].default),
            alpha=inputs.get("alpha", self.INPUTS["alpha"].default),
            dropout=inputs.get("dropout", self.INPUTS["dropout"].default),
            target_modules=inputs.get("target_modules"),
        )
        wrapper = ComfyUNetWrapper(
            weights.unet_sd,
            device=inputs.get("device", self.INPUTS["device"].default),
            dtype=inputs.get("dtype") or torch.bfloat16,
            use_checkpoint=inputs.get("use_checkpoint", self.INPUTS["use_checkpoint"].default),
            lora_config=lora_config,
        )
        result = {"model": ComfyUNetTrainableModel(wrapper)}
        self.validate_outputs(result)
        return result
