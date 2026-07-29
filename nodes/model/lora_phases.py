"""Multi-generation LoRA adapters and the node that grows a new one.

The feature this exists for: train a LoRA in phases (e.g. "warm-up" for
the first 20% of steps, then the rest), where each phase's own weight
changes stay separately extractable -- a checkpoint made from just the
post-warm-up phase should contain none of the warm-up phase's changes.

Design (see docs/vram_and_lora_phase_split.md for the full writeup,
including the alternative this rejected): stack a brand new, independent
adapter on top of the current one instead of merging the current one into
the frozen base weights and reinitializing it in place. Concretely, each
LoRAGeneration wraps an `inner` module (either the original
core.lora.LoRALinear/LoRAConv2d, or an earlier LoRAGeneration) and adds
its own low-rank delta on top, computed exactly like core.lora's own
layers (same math, reused via composition -- nothing here reimplements
LoRA math, it composes another instance of it):

    forward(x) = inner(x) + this_generation's_own_delta(x)

`inner`'s parameters are frozen (requires_grad_(False)) the moment a new
generation is stacked on it, so gradients only ever reach the newest
generation -- no base-weight mutation, no shared-tensor reinitialization,
and (the concrete reason this was chosen over merge()-then-reinit) no
optimizer-state entanglement: the new generation's lora_A/lora_B are
genuinely new tensors an OptimizerNode has never seen, so a fresh
optimizer for them starts with clean state, no manual reset needed.

Also here: pure functions to read weights back out, either one
generation's own contribution (extract_own_generation_weights, for "just
this phase") or every generation combined into one portable adapter
(extract_combined_weights, for "everything so far" -- what
ComfyUNetTrainableModel.trained_state_dict() calls). The combination is
exact, not approximate: concatenating every generation's (A, B*scaling)
pair along the rank axis produces a single rank-sum(r_i) adapter whose
output is bit-for-bit (up to ordinary floating-point summation order)
identical to the live stacked-generations forward pass. See
docs/vram_and_lora_phase_split.md for the derivation;
smoke_test_lora_phase_split.py checks it directly against a fresh
core.lora.LoRALinear/LoRAConv2d loaded with the combined weights, not
just against itself.

The LoRAGeneration classes are built lazily (_generation_classes(), first
call only) rather than at module import time, purely so importing this
module -- e.g. for LoRAPhaseSplitNode's INPUTS/OUTPUTS during graph
introspection -- never requires torch to be installed. Matches every
other nodes/ file's rule (see nodes_package_design.md: "the endpoint no
longer needs torch importable at all").
"""

from __future__ import annotations

from typing import ClassVar

from ..core import Node, Port
from .handle import TrainableModel, TrainedWeightsExportable
from .lora_injector import ComfyUNetTrainableModel

_generation_classes_cache = None


def _generation_classes():
    global _generation_classes_cache
    if _generation_classes_cache is not None:
        return _generation_classes_cache

    import math
    from abc import ABC, abstractmethod

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LoRAGeneration(nn.Module, ABC):
        """One trainable low-rank adapter stacked on top of a frozen
        `inner`. Subclasses only implement the shape-specific pieces
        (Linear vs. Conv2d); forward()/get_lora_weights() are shared here."""

        def __init__(self, inner: nn.Module, rank: int, alpha: float, dropout: float = 0.0):
            super().__init__()
            self.inner = inner
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self._build_params()

        @abstractmethod
        def _build_params(self) -> None:
            """Create self.lora_A / self.lora_B at this generation's own
            shape, inferred from self.inner's own lora_A/lora_B (works
            whether inner is a core.lora layer or an earlier generation --
            both expose that pair the same way)."""

        @abstractmethod
        def _delta(self, x):
            """This generation's own (already-scaled) contribution, same
            shape as inner(x)."""

        def forward(self, x):
            base = self.inner(x)
            return base + self._delta(x).to(base.dtype)

        def get_lora_weights(self):
            """Matches core.lora.LoRALinear/LoRAConv2d's own method name
            and return shape on purpose -- this generation's *own* raw
            (A, B), not combined with inner's."""
            return self.lora_A, self.lora_B

    class LinearLoRAGeneration(LoRAGeneration):

        def _build_params(self) -> None:
            in_features = self.inner.lora_A.shape[1]
            out_features = self.inner.lora_B.shape[0]
            device = self.inner.lora_A.device
            # fp32 regardless of the frozen chain's dtype -- same bf16
            # rounding-to-a-standstill reasoning as core.lora.LoRALinear.
            self.lora_A = nn.Parameter(torch.empty(self.rank, in_features,
                                                     device=device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank,
                                                     device=device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # lora_B stays zero -- this generation is a no-op at the
            # instant it's added, same as the very first generation was.

        def _delta(self, x):
            h = self.dropout(x).to(self.lora_A.dtype) @ self.lora_A.T
            return h @ (self.lora_B.T * self.scaling)

    class Conv2dLoRAGeneration(LoRAGeneration):

        def _build_params(self) -> None:
            rank, in_ch_per_group, kh, kw = self.inner.lora_A.shape
            out_channels = self.inner.lora_B.shape[0]
            device = self.inner.lora_A.device
            # Propagated from inner (present on both a core.lora.LoRAConv2d
            # and an earlier Conv2dLoRAGeneration) so an arbitrary-depth
            # stack always convolves with the same geometry as the
            # original layer.
            self.stride = self.inner.stride
            self.padding = self.inner.padding
            self.dilation = self.inner.dilation
            self.groups = self.inner.groups
            if self.rank % self.groups != 0:
                # The same constraint core.lora.LoRAConv2d's own first
                # conv2d(..., groups=self.groups) already has -- checked
                # here instead of leaving it to surface as a cryptic
                # "weight of size [...] instead" from inside F.conv2d.
                raise ValueError(
                    f"Conv2dLoRAGeneration: rank ({self.rank}) must be divisible by "
                    f"the layer's groups ({self.groups})."
                )
            self.lora_A = nn.Parameter(torch.empty(self.rank, in_ch_per_group, kh, kw,
                                                     device=device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_channels, self.rank, 1, 1,
                                                     device=device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def _delta(self, x):
            x_in = self.dropout(x).to(self.lora_A.dtype)
            adapter = F.conv2d(x_in, self.lora_A, None, self.stride, self.padding,
                                self.dilation, self.groups)
            # Second conv is always a plain 1x1 over the rank channels
            # (groups=1), independent of the original layer's own groups --
            # matches core.lora.LoRAConv2d.forward exactly.
            return F.conv2d(adapter, self.lora_B * self.scaling)

    _generation_classes_cache = (LoRAGeneration, LinearLoRAGeneration, Conv2dLoRAGeneration)
    return _generation_classes_cache


def lora_key(full_module_name: str) -> str:
    """Same convention as core.lora._key_to_lora_key -- this is a stable
    external file-format detail (ComfyUI/Kohya's lora_unet_... naming),
    not a design choice, so reimplementing the two-line mapping here
    (rather than importing a private helper across module boundaries) is
    just restating the same fixed format, not duplicating logic that
    could drift. Public (not _-prefixed): nodes/model/lora_checkpoint_loader.py
    needs the identical mapping too -- one shared copy in nodes/, not a
    third private reimplementation of the same two lines."""
    path = full_module_name
    if path.startswith("model.diffusion_model."):
        path = path[len("model.diffusion_model."):]
    return f"lora_unet_{path.replace('.', '_')}"


def _generation_chain(layer):
    """Oldest-first [(A, B, scaling), ...] for `layer` and everything it's
    stacked on top of, down to the original core.lora layer."""
    LoRAGeneration, _, _ = _generation_classes()
    chain = []
    node = layer
    while isinstance(node, LoRAGeneration):
        A, B = node.get_lora_weights()
        chain.append((A, B, node.scaling))
        node = node.inner
    A, B = node.get_lora_weights()
    chain.append((A, B, node.scaling))
    chain.reverse()
    return chain


def _combine_conv_generations(chain, groups: int):
    """Groups-aware version of the plain concatenation trick used for
    Linear. With groups > 1, a straight cat(dim=0) misaligns which
    output channels the grouped first conv2d associates with which input
    channels: PyTorch splits a grouped conv weight's output channels into
    `groups` *contiguous, equal-sized* blocks, one per input-channel
    group -- naively concatenating gen0's whole rank block then gen1's
    whole rank block re-carves those boundaries at the wrong points the
    moment rank0 != rank1 (or even when they're equal, the two
    generations' own internal group-blocks no longer line up with the
    concatenated tensor's group-blocks).

    Fix: reshape each generation's own A/B into (groups, its own
    rank_per_group, ...) -- i.e. its *own* group-blocks, made explicit --
    before concatenating along the rank_per_group axis and flattening
    back down. That makes the boundaries PyTorch's own automatic
    group-splitting will draw on the concatenated tensor land exactly on
    each generation's real group boundaries, by construction. For
    groups == 1 (the overwhelmingly common case for this project -- LoRA
    only targets nn.Conv2d at all when block_weights+target_all opts into
    it, and this architecture's convs aren't grouped) this reduces to
    exactly the plain concatenation, since there's only one group.
    """
    import torch
    A_by_group, B_by_group = [], []
    for A, B, scaling in chain:
        rank = A.shape[0]
        rank_per_group = rank // groups
        in_ch_per_group, kh, kw = A.shape[1], A.shape[2], A.shape[3]
        out_channels = B.shape[0]
        A_by_group.append(A.reshape(groups, rank_per_group, in_ch_per_group, kh, kw))
        B_by_group.append((B * scaling).reshape(out_channels, groups, rank_per_group, 1, 1))

    in_ch_per_group, kh, kw = A_by_group[0].shape[2:]
    out_channels = B_by_group[0].shape[0]
    A_cat = torch.cat(A_by_group, dim=1).reshape(-1, in_ch_per_group, kh, kw)
    B_cat = torch.cat(B_by_group, dim=2).reshape(out_channels, -1, 1, 1)
    return A_cat, B_cat


def extract_combined_weights(registry) -> dict:
    """Every generation folded into one portable (rank, alpha) adapter
    per layer. For a layer that's never been split (the common case),
    this is byte-identical to core.lora.extract_lora_weights's own
    output -- same tensors, same alpha -- so a model that never uses
    phase-splitting sees zero behavior change from this replacing the
    old extraction call in ComfyUNetTrainableModel.trained_state_dict().

    For an actually-combined (len > 1) chain: alpha is set equal to the
    combined rank so alpha/rank == 1 on reload -- each generation's own
    scaling is already baked into its B half below, so the combined pair
    needs an effective scaling of 1. This is what makes the file loadable
    by anything that infers rank from tensor shape and reads alpha at
    face value (this project's own loader included), not just something
    only this codebase understands. Conv2d layers go through
    _combine_conv_generations instead of a plain cat() -- see its
    docstring for why grouped convs need that.
    """
    import torch
    weights = {}
    for full_name, _parent, _attr, layer in registry:
        chain = _generation_chain(layer)
        key = lora_key(full_name)
        if len(chain) == 1:
            A, B, _scaling = chain[0]
            weights[f"{key}.lora_down.weight"] = A.detach().cpu().contiguous()
            weights[f"{key}.lora_up.weight"] = B.detach().cpu().contiguous()
            weights[f"{key}.alpha"] = torch.tensor([float(layer.alpha)], dtype=torch.float32)
            continue
        if chain[0][0].dim() == 4:
            A_cat, B_scaled_cat = _combine_conv_generations(chain, layer.groups)
        else:
            A_cat = torch.cat([A for A, _, _ in chain], dim=0)
            B_scaled_cat = torch.cat([B * s for _, B, s in chain], dim=1)
        weights[f"{key}.lora_down.weight"] = A_cat.detach().cpu().contiguous()
        weights[f"{key}.lora_up.weight"] = B_scaled_cat.detach().cpu().contiguous()
        weights[f"{key}.alpha"] = torch.tensor([float(A_cat.shape[0])], dtype=torch.float32)
    return weights


def extract_own_generation_weights(registry) -> dict:
    """Just the given registry's own layers' weights -- no inner chain
    walked. Called on a *snapshot* taken before a split swaps each
    layer's top-of-stack reference, so "own" means "the phase that just
    finished," not whatever's currently on top."""
    import torch
    weights = {}
    for full_name, _parent, _attr, layer in registry:
        A, B = layer.get_lora_weights()
        key = lora_key(full_name)
        weights[f"{key}.lora_down.weight"] = A.detach().cpu().contiguous()
        weights[f"{key}.lora_up.weight"] = B.detach().cpu().contiguous()
        weights[f"{key}.alpha"] = torch.tensor([float(layer.alpha)], dtype=torch.float32)
    return weights


def split_into_new_generation(wrapper, rank: int, alpha: float, dropout: float = 0.0) -> list:
    """Freeze every LoRA layer currently in `wrapper.lora_registry` and
    stack a fresh, independently-trainable generation on each. Mutates
    `wrapper` in place (swaps the live module-tree references via
    setattr, exactly like core.lora's own injection does, plus updates
    wrapper.lora_registry so lora_parameters()/get_lora_weights()/
    merge_lora() keep working against whatever's now on top).

    Returns the pre-split registry (the layers that just got frozen) --
    pass it to extract_own_generation_weights for a "just this phase"
    checkpoint before it goes out of scope.
    """
    if not wrapper.has_lora():
        raise ValueError("split_into_new_generation: model has no LoRA layers to split.")

    _, LinearLoRAGeneration, Conv2dLoRAGeneration = _generation_classes()

    frozen_snapshot = list(wrapper.lora_registry)
    new_registry = []
    for full_name, parent, attr_name, layer in wrapper.lora_registry:
        layer.lora_A.requires_grad_(False)
        layer.lora_B.requires_grad_(False)
        is_conv = layer.get_lora_weights()[0].dim() == 4
        generation_cls = Conv2dLoRAGeneration if is_conv else LinearLoRAGeneration
        new_layer = generation_cls(inner=layer, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, new_layer)
        new_registry.append((full_name, parent, attr_name, new_layer))
    wrapper.lora_registry = new_registry
    return frozen_snapshot


class FrozenLoRASnapshot(TrainedWeightsExportable):
    """A read-only, already-extracted LoRA state dict -- what
    LoRAPhaseSplitNode's `completed_generation` output actually is. No
    forward/training behavior, unlike TrainableModel; exists purely to be
    handed to LoRACheckpointSaverNode, which is exactly why this and
    TrainableModel share TrainedWeightsExportable instead of the saver
    depending on TrainableModel directly."""

    def __init__(self, state_dict: dict):
        self._state_dict = state_dict

    def trained_state_dict(self) -> dict:
        return self._state_dict


class LoRAPhaseSplitNode(Node):
    """Freezes the model's current LoRA adapter into an inert previous
    generation and starts a fresh one on top -- the "warm-up boundary"
    operation. Expresses a training curriculum (e.g. "20% warm-up, then
    the rest") structurally: wire two SupervisedLoRATrainerNode instances
    with this node between them, rather than adding a warmup_fraction
    flag to TrainerNode itself.

    The new generation's parameters are brand new tensors -- wire a
    *fresh* ModelParametersNode -> OptimizerNode -> TrainerNode after
    this node's `model` output; reusing the phase-1 optimizer would try
    to apply phase-1's accumulated momentum/second-moment state to
    weights it never saw.
    """

    OUTPUTS: ClassVar[dict] = {
        "model": Port(name="model", type=TrainableModel, required=True,
                       doc="The same model, now training from a fresh, empty adapter "
                           "stacked on top of the frozen previous one."),
        "completed_generation": Port(
            name="completed_generation", type=TrainedWeightsExportable, required=True,
            doc="Snapshot of just the phase that was frozen. Wire into its own "
                "LoRACheckpointSaverNode for a checkpoint containing only this phase's "
                "changes -- none of any earlier phase's."),
    }
    INPUTS: ClassVar[dict] = {
        "model": Port(name="model", type=TrainableModel, required=True),
        "rank": Port(name="rank", type=int, required=False, default=64,
                     doc="Rank of the new generation. Independent of the previous "
                         "generation's rank -- phases don't need to match."),
        "alpha": Port(name="alpha", type=float, required=False, default=1.0),
        "dropout": Port(name="dropout", type=float, required=False, default=0.0),
    }

    def build(self, **inputs) -> dict:
        self.validate_inputs(inputs)
        model = inputs["model"]
        if not isinstance(model, ComfyUNetTrainableModel):
            raise TypeError(
                f"LoRAPhaseSplitNode needs a ComfyUNetTrainableModel (operates on its "
                f"LoRA layer registry directly), got {type(model).__name__}."
            )
        frozen_snapshot = split_into_new_generation(
            model.raw,
            rank=inputs.get("rank", self.INPUTS["rank"].default),
            alpha=inputs.get("alpha", self.INPUTS["alpha"].default),
            dropout=inputs.get("dropout", self.INPUTS["dropout"].default),
        )
        completed = FrozenLoRASnapshot(extract_own_generation_weights(frozen_snapshot))
        result = {"model": model, "completed_generation": completed}
        self.validate_outputs(result)
        return result
