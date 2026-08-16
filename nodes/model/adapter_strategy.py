"""AdapterStrategy: how a trainable delta composes with a frozen weight.
See docs/training_pipeline_design.md section 3.1 for design rationale.

Plain LoRA -- a low-rank pair of matrices added to a frozen weight -- is
one way to parameterize a trainable delta, not the only one.
PlainLoRAAdapter wraps core.lora.LoRALinear/LoRAConv2d's math unchanged.

wrap() takes two parameters beyond a minimal (frozen, rank, scaling_policy)
signature, both structurally necessary: `alpha`, since
`scaling_policy.scaling(alpha, rank)` needs one, and `original` (the whole
source nn.Linear/nn.Conv2d), since LoRALinear/LoRAConv2d's constructors
need bias, in/out features, and conv stride/padding/dilation/groups --
not just a weight tensor.

PlainLoRAAdapter only honors BF16WeightStore, and checks that at wrap()
time rather than silently ignoring `frozen`. LoRALinear/LoRAConv2d's
forward() reads its own stored base_weight/base_bias buffers directly --
it never calls frozen.materialize(). For BF16WeightStore this is
behavior-identical to calling materialize() every forward pass
(materialize() is a no-op passthrough there), so wrapping the class
unchanged is correct. It would not be correct for a weight store whose
materialize() actually does work (e.g. dequantizing on the fly) --
honoring that needs a forward pass that actually calls materialize(),
which PlainLoRAAdapter's wrapped legacy class does not do.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .frozen_weight_store import BF16WeightStore, FrozenWeightStore
from .lora_scaling import LoRAScalingPolicy, _effective_alpha


class AdaptedLayer(ABC):
    """Structural contract for whatever AdapterStrategy.wrap() returns --
    a frozen weight combined with a trainable delta. Deliberately
    minimal: just the one member every current consumer
    (core.lora.extract_lora_weights, nodes/model/lora_phases.py) actually
    depends on by name. core.lora.LoRALinear/LoRAConv2d already satisfy
    this exactly (same method, same return shape) -- registered as
    virtual subclasses below via AdaptedLayer.register(), not by adding
    this as a base class in core/lora.py, so isinstance()/issubclass()
    checks work correctly without touching legacy code at all."""

    @abstractmethod
    def get_lora_weights(self):
        ...


class AdapterStrategy(ABC):
    """How a trainable delta composes with a frozen weight."""

    @abstractmethod
    def wrap(self, original, frozen: FrozenWeightStore, rank: int, alpha: float,
              scaling_policy: LoRAScalingPolicy, dropout: float = 0.0,
              weight: float = 1.0) -> AdaptedLayer:
        """`original`: the frozen nn.Linear/nn.Conv2d being adapted.
        `frozen`: that same layer's weight, already wrapped in a
        FrozenWeightStore -- an AdapterStrategy is free to reject a
        FrozenWeightStore kind it can't honor (see PlainLoRAAdapter).
        `weight`: core.lora's existing per-block scaling multiplier
        (LoRAConfig.block_weights), unrelated to LoRAScalingPolicy/
        rank/alpha -- kept as its own parameter to match what
        core.lora.LoRALinear/LoRAConv2d already take."""
        ...


class PlainLoRAAdapter(AdapterStrategy):
    """core.lora.LoRALinear/LoRAConv2d math, wrapped unchanged. See this
    module's docstring for the real limits this honors (BF16WeightStore
    only)."""

    def wrap(self, original, frozen: FrozenWeightStore, rank: int, alpha: float,
              scaling_policy: LoRAScalingPolicy, dropout: float = 0.0,
              weight: float = 1.0) -> AdaptedLayer:
        if not isinstance(frozen, BF16WeightStore):
            raise NotImplementedError(
                f"PlainLoRAAdapter only honors BF16WeightStore today, got "
                f"{type(frozen).__name__} -- see this class's own docstring, and "
                f"nodes/model/adapter_strategy.py's module docstring, for exactly why."
            )
        import torch.nn as nn

        LoRALinear, LoRAConv2d = _real_lora_classes()

        _register_legacy_adapted_layers()
        effective_alpha = _effective_alpha(alpha=alpha, rank=rank, policy=scaling_policy)
        if isinstance(original, nn.Linear):
            return LoRALinear(original, rank=rank, alpha=effective_alpha,
                               dropout=dropout, weight=weight)
        if isinstance(original, nn.Conv2d):
            return LoRAConv2d(original, rank=rank, alpha=effective_alpha,
                               dropout=dropout, weight=weight)
        raise TypeError(
            f"PlainLoRAAdapter.wrap(): original must be nn.Linear or nn.Conv2d, "
            f"got {type(original).__name__}."
        )


def _register_legacy_adapted_layers():
    """AdaptedLayer.register(core.lora.LoRALinear/LoRAConv2d) -- virtual
    subclass registration, so isinstance(layer, AdaptedLayer) is True for
    what PlainLoRAAdapter.wrap() actually returns, without core/lora.py
    declaring AdaptedLayer as a base class at all. Called from wrap()
    itself (idempotent -- ABCMeta.register() tolerates being called more
    than once) rather than at this module's import time, matching this
    project's rule that importing a nodes/ module for graph introspection
    alone should never require torch importable (see
    nodes/model/lora_phases.py's docstring for the same reasoning) --
    core.lora imports torch at its own module level, so this genuinely
    can't run until something -- here, wrap() -- actually needs it."""
    LoRALinear, LoRAConv2d = _real_lora_classes()
    AdaptedLayer.register(LoRALinear)
    AdaptedLayer.register(LoRAConv2d)


# Populated by nodes/model/adapter_injection.py's adapter_strategy_scope,
# at the one moment core.lora.LoRALinear/LoRAConv2d are guaranteed to
# still be the real classes -- right before it patches them. Never
# written to from this module.
_real_lora_classes_cache: dict = {}


def _real_lora_classes():
    """The real core.lora.LoRALinear/LoRAConv2d -- guaranteed real even
    while adapter_strategy_scope has core.lora.LoRALinear/LoRAConv2d
    patched to something else.

    Why this exists: PlainLoRAAdapter.wrap() has to construct the real
    classes regardless of what's calling it -- including when it's
    being used as the delegate inside some *other* AdapterStrategy
    (e.g. a future DoRAAdapter reusing PlainLoRAAdapter's construction
    for its base layer). If wrap() re-imported `core.lora.LoRALinear`
    live at that point, it would resolve to whatever
    adapter_strategy_scope currently has installed -- itself, in the
    case where PlainLoRAAdapter is the delegate -- recursing forever.
    Confirmed by hitting exactly that RecursionError while building the
    equivalence test for this (nodes/smoke_tests/smoke_test_adapter_injection.py),
    not theorized in advance.

    Falls back to a live import when the cache is empty, which is
    correct precisely because adapter_strategy_scope always populates
    the cache itself, at the one point core.lora's classes are still
    guaranteed real, before ever patching them -- so an empty cache
    means core.lora has never been patched at all yet, and its current
    classes are simply the real ones."""
    if _real_lora_classes_cache:
        return _real_lora_classes_cache["LoRALinear"], _real_lora_classes_cache["LoRAConv2d"]
    from core.lora import LoRAConv2d, LoRALinear
    return LoRALinear, LoRAConv2d
