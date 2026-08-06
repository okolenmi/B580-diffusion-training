"""AdapterStrategy: how a trainable delta composes with a frozen weight.
docs/training_pipeline_design.md section 3.1.

Plain LoRA -- a low-rank pair of matrices added to a frozen weight -- is
one way to parameterize a trainable delta, not the only one. Today's
core.lora-wrapping code hardcodes it as the only option; this makes it an
explicit choice. PlainLoRAAdapter below is that choice, made real and
tested; DoRAAdapter (Liu et al., "DoRA: Weight-Decomposed Low-Rank
Adaptation", arXiv:2402.09353, ICML 2024 Oral) is deliberately NOT built
here -- the design doc's own calibration for this section calls it "a
genuine new forward-pass code path (weight normalization + magnitude
scaling), not a formula tweak -- worth building... once the seam exists,
additive to PlainLoRAAdapter, not a replacement for it." This module is
that seam.

**Deviations from the design doc's minimal illustrative signature, and
why.** The doc shows `AdapterStrategy.wrap(self, frozen, rank,
scaling_policy) -> AdaptedLayer`. Two real gaps had to be closed to make
that actually callable:

1. No `alpha` parameter, despite `scaling_policy.scaling(alpha, rank)`
   structurally needing one -- added here as a required parameter.
2. `frozen: FrozenWeightStore` wraps one weight *tensor*
   (nodes/model/frozen_weight_store.py), but core.lora.LoRALinear/
   LoRAConv2d's constructors -- the "genuinely-correct legacy math"
   PlainLoRAAdapter wraps, unchanged, per this project's standing rule --
   need the *whole* original nn.Linear/nn.Conv2d (bias, in/out features,
   or conv stride/padding/dilation/groups), not just its weight. Added
   `original` as a required parameter alongside `frozen`.

**The more fundamental limit, stated plainly rather than glossed over:**
PlainLoRAAdapter only actually honors BF16WeightStore, and checks that at
wrap() time instead of silently ignoring `frozen`. core.lora.LoRALinear/
LoRAConv2d's forward() reads its own stored base_weight/base_bias buffers
directly -- it never calls frozen.materialize() at all. For
BF16WeightStore this is genuinely behavior-identical to "materialize()
every forward pass" (materialize() is a no-op passthrough there -- bf16
is already the working precision, nothing to dequantize), so wrapping the
legacy class unchanged is exactly correct. It would NOT be correct for a
real NF4WeightStore: honoring that for real needs the adapted layer's
forward pass to actually call frozen.materialize() every time (to
dequantize on the fly), which means either a genuinely new AdaptedLayer
implementation or a core/lora.py change -- neither of which is this
slice's job (NF4WeightStore itself is explicitly deferred, see
nodes/model/frozen_weight_store.py). `frozen` stays a real, checked
parameter here -- not a decorative one -- specifically so a future
NF4-aware AdapterStrategy has an actual, already-proven contract to
implement against, and so PlainLoRAAdapter fails loudly instead of
silently producing wrong results if handed a weight store it can't
actually honor.

**Also stated plainly: this is not yet wired into ComfyUNetLoRANode's
real construction path.** core.lora's actual per-layer injection lives
inside a tree-walk (core.lora._inject_lora) matching SDXL's specific
attention-block structure (to_q/to_k/to_v/to_out.0, time_embed/label_emb
segment matching, per-block weighting) -- genuinely complex, correct,
tested legacy logic. Wiring AdapterStrategy into it for real would mean
either modifying core/lora.py (against this project's standing rule) or
re-deriving that tree-walk in nodes/ (large, high-risk, arguably beyond
"medium effort"). Neither is this slice's job either -- this module
builds and equivalence-tests the seam itself
(smoke_test_adapter_strategy.py), which is what the backlog item actually
promises ("seam only... this slice is the seam, not the new techniques it
enables"). Live-wiring into the whole-UNet injection path is real,
separate, later work.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .frozen_weight_store import BF16WeightStore, FrozenWeightStore
from .lora_injector import LoRAScalingPolicy, _effective_alpha


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
    """Today's core.lora.LoRALinear/LoRAConv2d math, wrapped -- unchanged,
    per the existing rule: genuinely-correct legacy math gets wrapped,
    not re-derived. See this module's docstring for the real limits this
    honors rather than glosses over (BF16WeightStore only; not yet
    reachable from ComfyUNetLoRANode's real construction path)."""

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

        from core.lora import LoRAConv2d, LoRALinear

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
    from core.lora import LoRAConv2d, LoRALinear
    AdaptedLayer.register(LoRALinear)
    AdaptedLayer.register(LoRAConv2d)
