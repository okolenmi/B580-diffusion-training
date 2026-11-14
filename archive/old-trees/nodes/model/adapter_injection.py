"""adapter_layer_classes(): makes AdapterStrategy (adapter_strategy.py)
the thing ComfyUNetLoRANode actually builds with, not a separate,
equivalence-tested path next to the real construction code.

**This module used to monkeypatch.** It replaced `core.lora`'s
`LoRALinear`/`LoRAConv2d` names for the duration of one build, because
`core.lora._inject_lora` constructed its target layers by resolving those
two names from its own module namespace at call time -- so patching them
was the only way to change what the walk built. That worked, and it cost:

* **A concurrency hazard.** The names are module globals. Two
  `ComfyUNetLoRANode.build()` calls running at once would each patch the
  other's targets, and the docstring used to admit this outright.
* **`lora_class_cache.py`.** Because the names were rebound, anything
  that legitimately needed the *real* `LoRALinear` had to be handed them
  out of a side cache -- including strategies that recurse into
  themselves. That module and its cache exist only to defeat the patch.
* **An indirect cost.** `extract_lora_weights`/`load_lora_into_model`/
  `merge_lora_into_unet`/`lora_param_count` all gated on
  `isinstance(layer, (LoRALinear, LoRAConv2d))` against those same
  globals. Under a patch the gate silently answered the wrong question --
  the names then pointed at `_AdapterPatchedLayer`, whose `__new__`
  returns whatever the strategy built, so no real adapter was ever an
  instance of it. Every DoRA and NF4 layer was therefore skipped, which
  is the entire reason `reenable_dora_requires_grad()` and
  `dora_trainable_parameters()` below exist.

**None of that is needed now that `nodes/model/lora.py` is ours.**
`_inject_lora` takes the layer classes to build as an argument, so a
build states what it wants instead of arranging for something else to
look up the wrong thing. `adapter_layer_classes()` returns the pair to
pass; there is no scope to enter, nothing to restore, and nothing shared
between builds.

What survives unchanged is the interesting part: `AdapterStrategy.wrap()`
remains the single place that decides how a trainable delta composes with
a frozen weight, so the node graph and any other caller construct their
adapters through exactly the code the equivalence tests check.

**Alpha isn't double-applied.** ComfyUNetLoRANode.build() already
resolves scaling_policy into one effective alpha before injection
(lora_injector.py's `_effective_alpha()`) -- the `alpha` passed to each
target is already final. AdapterStrategy.wrap() takes its own
scaling_policy parameter and would apply it again if given the real one,
so the classes built below always pass `ClassicLoRAScaling()` (an
identity on an already-effective alpha), not whatever scaling_policy was
actually chosen.

**FrozenWeightStore is per-layer, not per-build.** It wraps
`original.weight`, which differs for every target, so the factory below
is called fresh per target layer rather than once per build.
"""

from __future__ import annotations

from .adapter_strategy import AdapterStrategy, PlainLoRAAdapter
from .frozen_weight_store import BF16WeightStore
from .lora_scaling import ClassicLoRAScaling


def adapter_layer_classes(adapter_strategy: AdapterStrategy,
                          frozen_weight_store_factory=None):
    """The `(linear_cls, conv_cls)` pair `inject_lora_into_unet` should
    build its targets with, routing every one through `adapter_strategy`.

    Both returned objects are the same class: its `__new__` never returns
    an instance of itself, so `__init__` never runs and callers get back
    exactly whatever `adapter_strategy.wrap()` returned -- a plain
    `LoRALinear`, a `DoRALinear`, an `NF4LoRALinear`, or whatever a
    future strategy returns. `_inject_lora`'s call site looks like an
    ordinary class construction and gets back an ordinary layer.

    One class serves both the linear and conv slots because
    `AdapterStrategy.wrap()` already dispatches on
    `isinstance(original, nn.Linear)` vs `nn.Conv2d` internally.

    frozen_weight_store_factory: a FrozenWeightStore class, or
    `(tensor) -> FrozenWeightStore` callable, invoked fresh per target
    layer since each layer's frozen weight is its own tensor. None (the
    default) means BF16WeightStore."""
    if frozen_weight_store_factory is None:
        frozen_weight_store_factory = BF16WeightStore

    class _AdapterLayer:
        def __new__(cls, original, rank: int = 64, alpha: float = 1.0,
                    dropout: float = 0.0, weight: float = 1.0):
            frozen = frozen_weight_store_factory(original.weight)
            return adapter_strategy.wrap(original, frozen, rank, alpha,
                                         ClassicLoRAScaling(), dropout, weight)

    return (_AdapterLayer, _AdapterLayer)


def reenable_dora_requires_grad(registry) -> None:
    """`ComfyUNetWrapper._init_lora()` (unet_wrapper.py) freezes every
    model parameter, then re-enables requires_grad on each LoRA layer's
    own lora_A/lora_B, gated by `hasattr(layer, "lora_A")`. That gate is
    False for a DoRALinear/DoRAConv2d (dora_layer.py) -- lora_A/lora_B
    live nested one level down (self._lora.lora_A), by composition, not
    inheritance. So the freeze runs, and the re-enable step does nothing
    for a DoRA layer: lora_A, lora_B, and magnitude (which _init_lora
    doesn't know about at all) all end up requires_grad=False.

    This corrects it from the outside -- called once, right after
    ComfyUNetWrapper's construction finishes, for every DoRA layer in the
    registry.

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
    """`ComfyUNetWrapper.lora_parameters()` has the same
    hasattr(layer, "lora_A") gate as reenable_dora_requires_grad() --
    see that function's docstring. It returns an empty pair for a
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