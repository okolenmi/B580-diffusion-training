"""ComfyUNetLoRANode: builds comfy's SDXL UNet from raw weights, injects LoRA.

Adapter only -- core.unet_wrapper.ComfyUNetWrapper and core.lora already do
the real work (UNet construction, LoRA layer injection); this wraps them
behind the TrainableModel/LoRAInjectorNode contracts.

LoRAScalingPolicy (docs/training_pipeline_design.md section 3.2): standard
LoRA scales its output by alpha/rank, which Kalajdzievski (arXiv:2312.03732,
2023) proves collapses adapter output/gradient magnitude as rank grows --
the reason LoRA is usually kept at low rank in practice, since higher ranks
"should" add capacity but empirically don't, because the scaling itself
suppresses them. RankStabilizedScaling's alpha/sqrt(rank) fixes that, at
zero VRAM/inference cost.

core.lora.LoRALinear/LoRAConv2d always compute `scaling = (alpha/rank) *
weight` themselves, internally, with no seam to override that formula
directly (confirmed by reading core/lora.py -- not modified here, per this
project's standing rule about legacy code). So this Node applies whatever
LoRAScalingPolicy it's given *before* core.lora ever runs, by solving for
the "effective alpha" that makes core.lora's own fixed formula land on the
policy's chosen scaling: `effective_alpha = policy.scaling(alpha, rank) *
rank`, so that core.lora's `effective_alpha / rank` recovers exactly
`policy.scaling(alpha, rank)`. For ClassicLoRAScaling this is an identity
(effective_alpha == alpha, zero behavior change); the algebra is otherwise
straightforward but worth being explicit about since it's the whole trick.

One real, worth-knowing side effect of this seam: core.lora.extract_lora_weights
saves each layer's *effective* alpha into the checkpoint (not the nominal
value this Node was given), and load_lora_into_model restores scaling via
its own hardcoded alpha/rank on resume -- which is exactly why the round
trip stays correct regardless of which policy produced the effective value
in the first place. The one cosmetic consequence: if a checkpoint's saved
alpha is inspected directly, or if core.lora's "alpha mismatch" print ever
fires on a config change, the number shown is the effective alpha, not
whatever nominal `alpha` was typed into this Node's Port.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from ..core import Port
from .handle import ModelWeights, TrainableModel
from .node import LoRAInjectorNode


class LoRAScalingPolicy(ABC):

    @abstractmethod
    def scaling(self, alpha: float, rank: int) -> float:
        ...


class ClassicLoRAScaling(LoRAScalingPolicy):
    """Today's actual behavior -- core.lora's existing alpha/rank formula,
    unchanged. Default, so nothing wired to this Node today changes."""

    def scaling(self, alpha: float, rank: int) -> float:
        return alpha / rank


class RankStabilizedScaling(LoRAScalingPolicy):
    """Kalajdzievski, "A Rank Stabilization Scaling Factor for Fine-Tuning
    with LoRA" (arXiv:2312.03732, 2023). Its actual value depends on
    training at higher rank than this project's current default (rank=64)
    to have anything to stabilize -- worth pairing with a rank increase,
    not independently useful at the current default rank by itself."""

    def scaling(self, alpha: float, rank: int) -> float:
        return alpha / (rank ** 0.5)


def _effective_alpha(alpha: float, rank: int, policy: LoRAScalingPolicy) -> float:
    """The seam itself, pulled out as its own function so it's directly
    testable without constructing a whole ComfyUNetWrapper -- see this
    module's docstring for the derivation. ClassicLoRAScaling is the
    identity (returns alpha unchanged); anything else changes what
    core.lora ends up computing for `scaling` without core.lora itself
    changing at all."""
    return policy.scaling(alpha, rank) * rank


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

    def trained_state_dict(self) -> dict:
        """Full-stack LoRA weights: every nodes/model/lora_phases.py
        generation this model has been split into, folded into one
        portable adapter (see that module's extract_combined_weights).
        Deferred import to avoid a lora_phases <-> lora_injector cycle --
        lora_phases already needs ComfyUNetTrainableModel at import time
        (LoRAPhaseSplitNode's isinstance check), so this direction has
        to be the lazy one. Identical output to a plain wrapper.get_lora_weights()
        call for a model that's never been split (single-generation registry)."""
        from .lora_phases import extract_combined_weights
        return extract_combined_weights(self._wrapper.lora_registry)

    @property
    def raw(self):
        """Escape hatch to the wrapped ComfyUNetWrapper, for callers (e.g.
        LoRAPhaseSplitNode) that need the full legacy object, not just
        the TrainableModel contract."""
        return self._wrapper


class ComfyUNetLoRANode(LoRAInjectorNode):

    INPUTS: ClassVar[dict[str, Port]] = {
        **LoRAInjectorNode.COMMON_INPUTS,
        "device": Port(name="device", type=str, required=False, default="xpu"),
        "dtype": Port(name="dtype", type=Any, required=False, default=None,
                       doc="torch dtype; None resolves to torch.bfloat16."),
        "rank": Port(name="rank", type=int, required=False, default=64),
        "alpha": Port(name="alpha", type=float, required=False, default=1.0),
        "scaling_policy": Port(
            name="scaling_policy", type=LoRAScalingPolicy, required=False, default=None,
            doc="None = ClassicLoRAScaling (today's exact alpha/rank behavior). "
                "RankStabilizedScaling (alpha/sqrt(rank)) is worth pairing with a higher "
                "rank than this Node's default (64) -- see LoRAScalingPolicy's module "
                "docstring for why alpha/rank alone suppresses higher ranks. Applied via "
                "an effective-alpha seam ahead of core.lora, not a core.lora change -- "
                "same docstring covers the one cosmetic side effect (checkpoint-saved "
                "alpha is the effective value, not this Port's nominal one).",
        ),
        "dropout": Port(name="dropout", type=float, required=False, default=0.0),
        "target_modules": Port(name="target_modules", type=Any, required=False, default=None),
        "use_checkpoint": Port(
            name="use_checkpoint", type=bool, required=False, default=True,
            doc="Gradient (activation) checkpointing -- trades recompute time for a real "
                "cut in peak VRAM (dominant cost is activations, not the frozen base weights "
                "or the tiny LoRA adapters). Defaults to True: VRAM is the priority here, and "
                "this is the single biggest lever available for it. Set False if you'd rather "
                "trade back for faster steps. Was previously unusable for LoRA -- ComfyUI's "
                "own checkpoint() (comfy/ldm/modules/diffusionmodules/util.py) passes an "
                "entire block's self.parameters() into torch.autograd.grad()'s inputs= list "
                "unfiltered, and a frozen base + LoRA block almost always has at least one "
                "frozen parameter (a norm weight, a bias, anything LoRA didn't target) in "
                "there, which used to crash with 'One of the differentiated Tensors does not "
                "require grad'. This node now patches that (nodes/model/gradient_checkpointing.py) "
                "before building the model whenever this is True -- see "
                "docs/vram_and_lora_phase_split.md for the root-cause writeup.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)
        import torch
        from core.lora import LoRAConfig
        from core.unet_wrapper import ComfyUNetWrapper

        weights: ModelWeights = inputs["weights"]
        use_checkpoint = inputs.get("use_checkpoint", self.INPUTS["use_checkpoint"].default)
        if use_checkpoint:
            from .gradient_checkpointing import enable_frozen_param_safe_checkpointing
            enable_frozen_param_safe_checkpointing()
        lora_config = LoRAConfig(
            rank=inputs.get("rank", self.INPUTS["rank"].default),
            alpha=_effective_alpha(
                alpha=inputs.get("alpha", self.INPUTS["alpha"].default),
                rank=inputs.get("rank", self.INPUTS["rank"].default),
                policy=inputs.get("scaling_policy") or ClassicLoRAScaling(),
            ),
            dropout=inputs.get("dropout", self.INPUTS["dropout"].default),
            target_modules=inputs.get("target_modules"),
        )
        wrapper = ComfyUNetWrapper(
            weights.unet_sd,
            device=inputs.get("device", self.INPUTS["device"].default),
            dtype=inputs.get("dtype") or torch.bfloat16,
            use_checkpoint=use_checkpoint,
            lora_config=lora_config,
        )
        result = {"model": ComfyUNetTrainableModel(wrapper)}
        self.validate_outputs(result)
        return result
