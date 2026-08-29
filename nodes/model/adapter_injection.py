"""adapter_strategy_scope: makes AdapterStrategy (adapter_strategy.py)
the thing ComfyUNetLoRANode actually builds with, not a separate,
equivalence-tested path next to the real construction code.

**The mechanism.** core.lora._inject_lora's tree-walk (which
nn.Linear/nn.Conv2d gets adapted, target-name matching, block weights)
is real and is not modified or duplicated here (core/ is read-only
reference material). But _inject_lora constructs its target layers by
calling `LoRALinear(child, ...)`/`LoRAConv2d(...)` as plain,
module-level names, resolved from core.lora's own namespace at call
time, not at import time. Temporarily replacing what those two names
point to changes what _inject_lora builds at each target, without
touching core/lora.py.

**Why this is a scope, not a permanent patch.** A later,
ComfyUNetLoRANode.build() call in the same process (a different graph,
a different adapter_strategy, or none) needs core.lora.LoRALinear/
LoRAConv2d back to their real selves -- leaving the patch installed
would corrupt every subsequent LoRA build. adapter_strategy_scope() is
a context manager so restoration happens on every exit, exception or
not, scoped to one ComfyUNetWrapper(...) construction call.

**Why PlainLoRAAdapter + BF16WeightStore installs no patch at all.**
PlainLoRAAdapter.wrap() (adapter_strategy.py) constructs LoRALinear/
LoRAConv2d by importing them fresh from core.lora inside its own
wrap() body. Patching those same names and then routing
PlainLoRAAdapter through the patched path would make that import
resolve to the patch it's running inside of -- infinite recursion.
When frozen_weight_store_factory is BF16WeightStore (the default),
PlainLoRAAdapter *is* core.lora's own unmodified behavior, so skipping
the patch is simply correct, not an optimization -- and it's the
default (adapter_strategy=None -> PlainLoRAAdapter(),
frozen_weight_store_factory=None -> BF16WeightStore), so nothing wired
to ComfyUNetLoRANode today is affected.

That stopped covering every real case once PlainLoRAAdapter also
learned to honor NF4WeightStore (adapter_strategy.py):
adapter_strategy=None with frozen_weight_store_factory=NF4WeightStore
is a real combination (NF4-quantized base, plain LoRA on top), and
skipping the patch for it would do nothing -- core.lora's real
LoRALinear/LoRAConv2d always capture a bf16 base_weight buffer
directly, with no NF4 concept. The skip condition below checks both
PlainLoRAAdapter *and* BF16WeightStore, not PlainLoRAAdapter alone.
No recursion risk when NF4WeightStore is selected while patched:
PlainLoRAAdapter.wrap()'s NF4 branch constructs nf4_lora_layer.py's
NF4LoRALinear/NF4LoRAConv2d directly, never touching core.lora.LoRALinear/
LoRAConv2d.

**Alpha isn't double-applied.** ComfyUNetLoRANode.build() already
resolves scaling_policy into one effective alpha before core.lora runs
(lora_injector.py's _effective_alpha()) -- the `alpha` _inject_lora
passes to each target is already final. AdapterStrategy.wrap() takes
its own scaling_policy parameter and would apply it again if given the
real one, so the patched class below always passes
ClassicLoRAScaling() (an identity on an already-effective alpha), not
whatever scaling_policy was actually chosen.

**A second recursion risk.** PlainLoRAAdapter.wrap() can be called
indirectly -- a future AdapterStrategy might reuse its LoRALinear/
LoRAConv2d construction as a base layer. If PlainLoRAAdapter.wrap()
re-imported core.lora.LoRALinear live at that point, it would resolve
to whatever's currently patched, recursing. Fixed by caching the real
classes (adapter_strategy.py's `_real_lora_classes()`, via
lora_class_cache.py) at the one moment they're guaranteed real --
before patching -- so PlainLoRAAdapter.wrap() always gets the real
ones regardless of what's currently patched.

**Concurrency.** core.lora.LoRALinear/LoRAConv2d are shared module
globals for the duration of this scope; a second ComfyUNetLoRANode.build()
running concurrently on another thread would race with this one.
Construction was never documented as thread-safe; this is the first
place that assumption is load-bearing rather than incidental.
"""

from __future__ import annotations

import contextlib

from .adapter_strategy import AdapterStrategy, PlainLoRAAdapter
from .frozen_weight_store import BF16WeightStore
from .lora_class_cache import _real_lora_classes_cache
from .lora_scaling import ClassicLoRAScaling


def _make_adapter_patched_class(adapter_strategy: AdapterStrategy, frozen_weight_store_factory):
    """A class whose __new__ never returns an instance of itself, so
    __init__ never runs -- callers get back exactly whatever
    adapter_strategy.wrap() returned (a LoRALinear, LoRAConv2d, or any
    other AdaptedLayer), not an instance of this class. This is what
    makes `core.lora.LoRALinear = this` work: _inject_lora's
    `LoRALinear(child, rank=..., ...)` call looks like an ordinary
    class call and gets back an ordinary layer object.

    One class serves both the LoRALinear and LoRAConv2d slots --
    AdapterStrategy.wrap() already dispatches on isinstance(original,
    nn.Linear) vs. nn.Conv2d internally.

    frozen_weight_store_factory: a FrozenWeightStore class (or (tensor)
    -> FrozenWeightStore callable), called fresh per target layer, since
    each layer's frozen weight is its own tensor."""

    class _AdapterPatchedLayer:
        def __new__(cls, original, rank: int = 64, alpha: float = 1.0,
                    dropout: float = 0.0, weight: float = 1.0):
            frozen = frozen_weight_store_factory(original.weight)
            return adapter_strategy.wrap(original, frozen, rank, alpha,
                                          ClassicLoRAScaling(), dropout, weight)

    return _AdapterPatchedLayer


@contextlib.contextmanager
def adapter_strategy_scope(adapter_strategy: AdapterStrategy, frozen_weight_store_factory=None):
    """Everything core.lora._inject_lora constructs inside this `with`
    block goes through adapter_strategy instead of core.lora's own
    LoRALinear/LoRAConv2d -- unless adapter_strategy is a
    PlainLoRAAdapter *and* frozen_weight_store_factory is BF16WeightStore
    (the default for both), in which case nothing is patched (see this
    module's docstring for why that combination is a real no-op).
    Restores core.lora's real classes on every exit, exception or not.

    frozen_weight_store_factory: a FrozenWeightStore class (or (tensor)
    -> FrozenWeightStore callable), or None for the default
    (BF16WeightStore)."""
    if frozen_weight_store_factory is None:
        frozen_weight_store_factory = BF16WeightStore

    if isinstance(adapter_strategy, PlainLoRAAdapter) and frozen_weight_store_factory is BF16WeightStore:
        yield
        return

    import core.lora as core_lora

    original_linear = core_lora.LoRALinear
    original_conv2d = core_lora.LoRAConv2d
    _real_lora_classes_cache["LoRALinear"] = original_linear
    _real_lora_classes_cache["LoRAConv2d"] = original_conv2d
    patched = _make_adapter_patched_class(adapter_strategy, frozen_weight_store_factory)
    core_lora.LoRALinear = patched
    core_lora.LoRAConv2d = patched
    try:
        yield
    finally:
        core_lora.LoRALinear = original_linear
        core_lora.LoRAConv2d = original_conv2d


def reenable_dora_requires_grad(registry) -> None:
    """core.unet_wrapper.ComfyUNetWrapper._init_lora() (frozen legacy
    code, runs inside adapter_strategy_scope's `with` block above)
    freezes every model parameter, then re-enables requires_grad on
    each LoRA layer's own lora_A/lora_B, gated by
    `hasattr(layer, "lora_A")`. That gate is False for a DoRALinear/
    DoRAConv2d (dora_layer.py) -- lora_A/lora_B live nested one level
    down (self._lora.lora_A), by composition, not inheritance. So the
    freeze runs, and the re-enable step does nothing for a DoRA layer:
    lora_A, lora_B, and magnitude (which _init_lora doesn't know about
    at all) all end up requires_grad=False.

    core/unet_wrapper.py is frozen, so this corrects it from the
    outside -- called once, right after ComfyUNetWrapper's construction
    finishes, for every DoRA layer in the registry.

    Without this, a DoRAAdapter-built model has zero trainable
    parameters in its DoRA layers: gradients for them are never
    computed, but nothing raises -- loss can still move from whatever
    else is trainable (e.g. a text-encoder LoRA), so the run looks
    normal unless requires_grad or the trainable-parameter count is
    checked directly."""
    from .dora_layer import DoRAConv2d, DoRALinear

    for _full_name, _parent, _attr, layer in registry:
        if isinstance(layer, (DoRALinear, DoRAConv2d)):
            A, B = layer.get_lora_weights()
            A.requires_grad_(True)
            B.requires_grad_(True)
            layer.magnitude.requires_grad_(True)


def dora_trainable_parameters(registry) -> list:
    """core.unet_wrapper.ComfyUNetWrapper.lora_parameters() has the
    same hasattr(layer, "lora_A") gate as reenable_dora_requires_grad()
    -- see that function's docstring. It returns an empty pair for a
    bare, top-of-stack DoRALinear/DoRAConv2d and never returns
    magnitude at all.

    This matters even with requires_grad set correctly:
    ComfyUNetTrainableModel.trainable_parameters() (lora_injector.py) is
    what actually builds the list handed to an optimizer, and an
    optimizer only steps on parameters it was given -- requires_grad
    only controls whether a gradient gets computed. Without this
    function, gradients for DoRA parameters would be computed but the
    optimizer would never receive them, so none would move.
    ComfyUNetTrainableModel.footprint_bytes() also uses
    lora_parameters() (via data_ptr()) to separate "trainable adapter"
    from "frozen base" for memory reporting -- without this, a DoRA
    layer's lora_A/lora_B/magnitude would be counted as part of the
    frozen base.

    Only a bare, top-of-stack DoRALinear/DoRAConv2d needs anything
    added here. A DoRA layer wrapped in a later LoRAGeneration
    (lora_phases.py, after a phase split) has its own direct
    lora_A/lora_B, which lora_parameters()'s hasattr check already
    matches (LoRAGeneration doesn't use composition) -- and that frozen
    layer's magnitude is correctly excluded here too, since
    split_into_new_generation() already set requires_grad=False on it."""
    from .dora_layer import DoRAConv2d, DoRALinear

    params = []
    for _full_name, _parent, _attr, layer in registry:
        if isinstance(layer, (DoRALinear, DoRAConv2d)):
            A, B = layer.get_lora_weights()
            params.append(A)
            params.append(B)
            params.append(layer.magnitude)
    return params
