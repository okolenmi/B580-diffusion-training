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

LoRAScalingPolicy/ClassicLoRAScaling/RankStabilizedScaling/_effective_alpha
now live in lora_scaling.py, re-exported here unchanged -- moved once
nodes/model/adapter_injection.py needed them too and importing from here
directly would have cycled back through this module. See lora_scaling.py's
own docstring.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..core import Port
from ..resource_policy import ResourcePolicy
from .frozen_weight_store import BF16WeightStore
from .adapter_strategy import AdapterStrategy
from .gradient_checkpointing import FrozenParamSafeCheckpointing, NoCheckpointing
from .handle import ModelWeights, TrainableModel
from .lora_scaling import ClassicLoRAScaling, LoRAScalingPolicy, RankStabilizedScaling, _effective_alpha
from .node import LoRAInjectorNode

__all__ = [
    "ClassicLoRAScaling", "LoRAScalingPolicy", "RankStabilizedScaling",
    "ComfyUNetTrainableModel", "ComfyUNetLoRANode",
]


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

    def footprint_bytes(self) -> int:
        """The frozen base's footprint -- everything in the wrapped
        UNetModel's state_dict() except the currently-trainable LoRA A/B
        parameters, each wrapped in BF16WeightStore (today's actual
        storage; see that class's docstring for why this doesn't change
        when a future NF4WeightStore lands). Matched against
        lora_parameters() by data_ptr(), not name or object identity:
        state_dict() calls .detach() on every tensor by default, a new
        Python object sharing the same underlying storage, so identity
        comparison against the live nn.Parameter objects would silently
        never match anything -- checked directly, not assumed. Name-based
        matching (anything ending in ".lora_A"/".lora_B") was considered
        and rejected: after a nodes/model/lora_phases.py phase split, an
        earlier, now-frozen generation's own lora_A/lora_B keeps that
        exact name at its own nested path (LoRAGeneration.inner is a
        genuine submodule), so a name-based filter would wrongly exclude
        already-frozen weight from the frozen count. data_ptr() against
        lora_parameters() -- which already correctly returns only the
        top-of-stack, currently-trainable pair per layer -- gets this
        right in both the split and non-split case for free."""
        if self._wrapper is None:
            return 0
        trainable_ptrs = {p.data_ptr() for p in self._wrapper.lora_parameters()}
        return sum(
            BF16WeightStore(tensor).footprint_bytes()
            for tensor in self._wrapper.state_dict().values()
            if tensor.data_ptr() not in trainable_ptrs
        )

    def offload(self) -> None:
        """Cheap, reversible: move to CPU, remember the device to come
        back to. Doesn't touch LoRA state, dtype, or anything about the
        model's identity -- same object, same weights, just relocated."""
        self._device_before_offload = self._wrapper.device
        self._wrapper.to(device="cpu")

    def reload(self, device: str | None = None) -> None:
        target = device or getattr(self, "_device_before_offload", None)
        if target is None:
            raise RuntimeError(
                "reload() needs an explicit device, or a prior offload() to "
                "remember one -- neither was given."
            )
        self._wrapper.to(device=target)

    def release(self) -> None:
        """Not reversible -- drops the wrapped model entirely. Moves to
        CPU first for a clean drop rather than releasing a live device
        reference, then drops the reference itself; whatever built this
        TrainableModel has to build it again. Every other method on this
        object (forward/train/eval/to/trained_state_dict) is correctly
        unusable afterward -- only footprint_bytes() is special-cased to
        report 0 rather than raise, matching DeviceResident's own
        "best-effort current usage" contract for a released object."""
        if self._wrapper is not None:
            self._wrapper.to(device="cpu")
            self._wrapper = None
        import gc
        gc.collect()


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
                "or the tiny LoRA adapters). Defaults to True. Set False to trade back for "
                "faster steps. Mapped internally to FrozenParamSafeCheckpointing (True) or "
                "NoCheckpointing (False) -- see nodes/model/gradient_checkpointing.py. "
                "Ignored if resource_policy is given.",
        ),
        "resource_policy": Port(
            name="resource_policy", type=ResourcePolicy, required=False, default=None,
            doc="None = use_checkpoint and scaling_policy above, independently. When given, "
                "fully replaces both -- its checkpointing_strategy() and "
                "lora_scaling_policy() are used instead, and use_checkpoint/scaling_policy "
                "are ignored.",
        ),
        "adapter_strategy": Port(
            name="adapter_strategy", type=AdapterStrategy, required=False, default=None,
            doc="None = PlainLoRAAdapter (today's exact core.lora.LoRALinear/LoRAConv2d "
                "behavior, byte-identical -- nothing patched at all, see "
                "nodes/model/adapter_injection.py for why that's correct, not a shortcut). "
                "Any other AdapterStrategy (nodes/model/adapter_strategy.py) is live-wired "
                "into core.lora's real, unmodified injection tree-walk via a scoped, "
                "restored-on-exit patch -- see adapter_injection.py's module docstring for "
                "the full mechanism and the alpha-double-application pitfall it avoids.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainableModel]:
        self.validate_inputs(inputs)
        import torch
        from core.lora import LoRAConfig
        from core.unet_wrapper import ComfyUNetWrapper

        from .adapter_injection import adapter_strategy_scope
        from .adapter_strategy import PlainLoRAAdapter

        weights: ModelWeights = inputs["weights"]
        adapter_strategy = inputs.get("adapter_strategy") or PlainLoRAAdapter()
        resource_policy = inputs.get("resource_policy")
        if resource_policy is not None:
            checkpointing_strategy = resource_policy.checkpointing_strategy()
            scaling_policy = resource_policy.lora_scaling_policy()
            use_checkpoint = not isinstance(checkpointing_strategy, NoCheckpointing)
        else:
            use_checkpoint = inputs.get("use_checkpoint", self.INPUTS["use_checkpoint"].default)
            checkpointing_strategy = (
                FrozenParamSafeCheckpointing() if use_checkpoint else NoCheckpointing())
            scaling_policy = inputs.get("scaling_policy") or ClassicLoRAScaling()
        checkpointing_strategy.apply()
        lora_config = LoRAConfig(
            rank=inputs.get("rank", self.INPUTS["rank"].default),
            alpha=_effective_alpha(
                alpha=inputs.get("alpha", self.INPUTS["alpha"].default),
                rank=inputs.get("rank", self.INPUTS["rank"].default),
                policy=scaling_policy,
            ),
            dropout=inputs.get("dropout", self.INPUTS["dropout"].default),
            target_modules=inputs.get("target_modules"),
        )
        with adapter_strategy_scope(adapter_strategy):
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
