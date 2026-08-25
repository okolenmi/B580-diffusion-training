"""adapter_strategy_scope: makes AdapterStrategy (adapter_strategy.py)
the real thing ComfyUNetLoRANode builds, not just an equivalence-tested
seam sitting next to the actual construction path.
See docs/training_pipeline_design.md section 3.1/10 for the rationale.

Also home to reenable_dora_requires_grad() -- a second, independent
real gap in the same underlying territory (core/'s frozen assumption
that a LoRA layer's lora_A/lora_B are its own direct attributes, not
true for a DoRALinear/DoRAConv2d), found and closed while wiring in
DoRA's checkpoint round-trip (design doc section 3.1). See that
function's own docstring for the real, previously-silent bug this
closes: without it, a DoRAAdapter-built model trains nothing in its
adapted layers at all.

**The constraint this works around.** core.lora._inject_lora's tree-walk
(which nn.Linear/nn.Conv2d gets adapted -- to_q/to_k/to_v/to_out.0,
time_embed/label_emb segment matches, block_weights) is real, delicate,
already-correct logic this project's own standing rule says not to
modify or duplicate (core/ is read-only reference material -- see this
project's "wrap-don't-copy" rule, and core.lora.py's own module
docstring). But _inject_lora constructs its target layers by calling
`LoRALinear(child, rank=..., alpha=..., dropout=..., weight=...)` and
`LoRAConv2d(...)` as plain, undecorated calls to module-level names --
Python resolves those names from core.lora's own namespace at call
time, not at core.lora's import time. That's a real, exploitable seam:
temporarily replacing what those two names point to changes what
_inject_lora's unmodified targeting logic actually builds at each
target it finds, without touching a single line of core/lora.py.

**Why this is a scope, not a permanent patch (unlike
gradient_checkpointing.py's).** That patch is idempotent-permanent by
design -- once frozen-param-safe checkpointing is correct, it should
just stay correct for the rest of the process. This one is different:
a later, unrelated ComfyUNetLoRANode.build() call in the same process
(a different graph, a different adapter_strategy, or none at all) needs
core.lora.LoRALinear/LoRAConv2d back to their real selves -- leaving
this patch installed permanently would silently corrupt every LoRA
build after the first one that used a non-default adapter_strategy.
adapter_strategy_scope() is a context manager specifically so
restoration happens on every exit, including via an exception, and
scopes to exactly the one ComfyUNetWrapper(...) construction call that
needs it.

**Why PlainLoRAAdapter installs no patch at all -- but only when it
really would be a no-op.** PlainLoRAAdapter.wrap() (adapter_strategy.py)
itself constructs LoRALinear/LoRAConv2d by importing them fresh from
core.lora inside its own wrap() body -- if this module patched those
same names and then routed PlainLoRAAdapter through the patched path,
PlainLoRAAdapter.wrap()'s own import would resolve to the patch it's
currently running inside of, recursing forever. There's a real,
structural reason this never comes up when `frozen_weight_store_factory`
is BF16WeightStore (the default): PlainLoRAAdapter *is* core.lora's own
unmodified behavior for that case, by definition, so there is nothing to
intercept -- skipping the patch entirely is not an optimization
shortcut, it's the only correct behavior, and it happens to also be the
default (adapter_strategy=None -> PlainLoRAAdapter(),
frozen_weight_store_factory=None -> BF16WeightStore), so nothing wired to
ComfyUNetLoRANode today changes at all.

**This stopped being universally true once PlainLoRAAdapter also learned
to honor NF4WeightStore** (adapter_strategy.py). Selecting
`adapter_strategy=None` (PlainLoRAAdapter) with
`frozen_weight_store_factory=NF4WeightStore` is a real, deliberate
combination -- NF4-quantized base, plain LoRA on top -- and skipping the
patch for it would silently do nothing: core.lora's own real
LoRALinear/LoRAConv2d always capture a bf16 base_weight buffer directly
from `original.weight`, with no NF4 concept at all. The skip condition
below is precise about this: PlainLoRAAdapter *and* the BF16WeightStore
factory, both, not PlainLoRAAdapter alone. No recursion risk either way
NF4WeightStore is selected while patched: PlainLoRAAdapter.wrap()'s
NF4WeightStore branch constructs nf4_lora_layer.py's NF4LoRALinear/
NF4LoRAConv2d directly, never touching core.lora.LoRALinear/LoRAConv2d
at all -- there's nothing for it to recurse through.

**The alpha double-application this module has to avoid.**
ComfyUNetLoRANode.build() already resolves scaling_policy into a single
*effective* alpha before core.lora ever runs (lora_injector.py's
_effective_alpha() seam, section 3.2) -- LoRAConfig.alpha, and therefore
the `alpha` _inject_lora passes to each target, is already the final,
policy-adjusted value by the time this module's patched class sees it.
AdapterStrategy.wrap() takes its own `scaling_policy` parameter and
would apply it *again* if given the real one -- so the patched class
below always passes ClassicLoRAScaling() (a proven identity on an
already-effective alpha, per _effective_alpha's own docstring), not
whatever scaling_policy the person actually chose. This is the correct
way to say "already applied upstream, treat this as final" using
machinery that already exists, not a special case invented for this
module -- but it is exactly the kind of thing a future adapter_strategy
implementation needs to know about the call convention here, so it's
recorded in one place, not left to be rediscovered.

**A second, deeper recursion risk this module has to prevent, beyond
skipping the patch for PlainLoRAAdapter.** Even for a different
adapter_strategy, PlainLoRAAdapter.wrap() can still end up called
*indirectly* -- naturally, if some future AdapterStrategy (a DoRAAdapter,
say) wants to reuse PlainLoRAAdapter's own LoRALinear/LoRAConv2d
construction as its base layer rather than reimplementing it. If
PlainLoRAAdapter.wrap() re-imported `core.lora.LoRALinear` live at that
point, it would resolve to whatever this module currently has patched in
-- recursing forever. Confirmed by hitting exactly that RecursionError
while building this module's own equivalence test, not theorized in
advance. Fixed by having this module cache the real classes (into
adapter_strategy.py's PlainLoRAAdapter, which imports it from
lora_class_cache.py directly) at the one moment
they're still guaranteed real -- right here, before patching -- so
PlainLoRAAdapter.wrap() (via adapter_strategy.py's `_real_lora_classes()`)
always gets the real ones regardless of what's currently patched or what
called it.

**Concurrency.** Relies on section 5.6's stated single-threaded
construction contract -- core.lora.LoRALinear/LoRAConv2d are shared
module globals for the duration of this scope; a second
ComfyUNetLoRANode.build() running concurrently on another thread would
race with this one. Not a new assumption this module introduces
(construction was never documented as thread-safe), but worth stating
here since this is the first place that assumption is load-bearing
rather than incidental.
"""

from __future__ import annotations

import contextlib

from .adapter_strategy import AdapterStrategy, PlainLoRAAdapter
from .frozen_weight_store import BF16WeightStore
from .lora_class_cache import _real_lora_classes_cache
from .lora_scaling import ClassicLoRAScaling


def _make_adapter_patched_class(adapter_strategy: AdapterStrategy, frozen_weight_store_factory):
    """A class whose __new__ never returns an instance of itself --
    per the language's own data model, that means __init__ never runs
    on the result, so what callers actually get back is exactly
    whatever adapter_strategy.wrap() returned (a real LoRALinear,
    LoRAConv2d, or any other AdaptedLayer), not an instance of this
    throwaway class. This is what makes `core.lora.LoRALinear = this`
    work: _inject_lora's `LoRALinear(child, rank=..., ...)` call sees
    an ordinary class call and gets back an ordinary layer object,
    with no idea anything was substituted.

    One class, used for both the LoRALinear and LoRAConv2d slots --
    AdapterStrategy.wrap() already dispatches on isinstance(original,
    nn.Linear) vs. nn.Conv2d internally (see PlainLoRAAdapter.wrap()),
    so there is nothing this factory needs to decide that wrap() isn't
    already deciding correctly.

    frozen_weight_store_factory: a real FrozenWeightStore class (or any
    (tensor) -> FrozenWeightStore callable) -- called fresh per target
    layer, since each layer's frozen weight is its own separate tensor,
    not something to share a single FrozenWeightStore instance across."""

    class _AdapterPatchedLayer:
        def __new__(cls, original, rank: int = 64, alpha: float = 1.0,
                    dropout: float = 0.0, weight: float = 1.0):
            frozen = frozen_weight_store_factory(original.weight)
            # ClassicLoRAScaling(), always -- see this module's docstring
            # for exactly why the real scaling_policy must not be passed
            # here.
            return adapter_strategy.wrap(original, frozen, rank, alpha,
                                          ClassicLoRAScaling(), dropout, weight)

    return _AdapterPatchedLayer


@contextlib.contextmanager
def adapter_strategy_scope(adapter_strategy: AdapterStrategy, frozen_weight_store_factory=None):
    """Everything core.lora._inject_lora constructs inside this `with`
    block goes through adapter_strategy instead of core.lora's own
    LoRALinear/LoRAConv2d -- unless adapter_strategy is a
    PlainLoRAAdapter *and* frozen_weight_store_factory is BF16WeightStore
    (the default for both), in which case nothing is patched at all (see
    this module's docstring for why that specific combination, and only
    that one, is a real no-op). Restores core.lora's real classes on
    every exit, exception or not.

    frozen_weight_store_factory: a FrozenWeightStore class (or (tensor)
    -> FrozenWeightStore callable), or None for the default
    (BF16WeightStore, today's exact behavior)."""
    if frozen_weight_store_factory is None:
        frozen_weight_store_factory = BF16WeightStore

    if isinstance(adapter_strategy, PlainLoRAAdapter) and frozen_weight_store_factory is BF16WeightStore:
        yield
        return

    import core.lora as core_lora

    original_linear = core_lora.LoRALinear
    original_conv2d = core_lora.LoRAConv2d
    # Cache the real classes before patching -- this is the one moment
    # they're guaranteed real. See this module's docstring for why
    # PlainLoRAAdapter.wrap() needs this rather than a live import.
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
    does exactly two things after real injection: freeze every model
    parameter (`for p in self.model.parameters(): p.requires_grad_(False)`),
    then re-enable requires_grad on each LoRA layer's own lora_A/lora_B,
    gated by `hasattr(layer, "lora_A")`. That gate is False for a
    DoRALinear/DoRAConv2d (dora_layer.py) -- that pair lives nested one
    level down (self._lora.lora_A), built via composition, not
    inheritance (see dora_layer.py's own module docstring for why; the
    same attribute-layout assumption separately broke
    lora_phases.py's split_into_new_generation -- see that function's
    own docstring for the sibling bug). So the blanket freeze runs, and
    the supposed-to-undo-it-for-LoRA-params step silently does nothing
    for a DoRA layer: lora_A, lora_B, AND magnitude (which _init_lora
    doesn't even know exists -- it predates DoRA entirely, DoRA's own
    trainable parameter isn't lora_A/lora_B at all) all end up
    requires_grad=False.

    core/unet_wrapper.py is frozen (this project's standing rule), so
    this corrects it from the outside instead -- called once, right
    after ComfyUNetWrapper's construction finishes (still inside
    ComfyUNetLoRANode.build(), before the model is handed back), for
    every DoRA layer currently in the registry. Same overall approach
    this module's own adapter_strategy_scope already takes for a
    different core/ assumption it can't change directly: work around
    it from nodes/, don't touch core/.

    A real, previously-silent bug this closes, found while wiring in
    DoRA's checkpoint round-trip (design doc section 3.1) and unrelated
    to it -- this is about training, not saving: without this call, a
    DoRAAdapter-built model has ZERO trainable parameters in every DoRA
    layer. A real training run through
    ComfyUNetLoRANode(adapter_strategy=DoRAAdapter()) would train
    nothing in the adapted layers at all -- no crash, no error,
    gradients for those parameters are simply never computed; loss
    would still move from whatever else is trainable (e.g. an unrelated
    text-encoder LoRA), with nothing about the run visibly signaling
    the problem short of checking requires_grad directly, or noticing
    the printed trainable-parameter count is far smaller than
    core.lora.lora_param_count()'s own report for the same registry.
    """
    from .dora_layer import DoRAConv2d, DoRALinear

    for _full_name, _parent, _attr, layer in registry:
        if isinstance(layer, (DoRALinear, DoRAConv2d)):
            A, B = layer.get_lora_weights()
            A.requires_grad_(True)
            B.requires_grad_(True)
            layer.magnitude.requires_grad_(True)


def dora_trainable_parameters(registry) -> list:
    """The other half of the same gap reenable_dora_requires_grad()
    closes (see that function's own docstring for the full derivation)
    -- core.unet_wrapper.ComfyUNetWrapper.lora_parameters() (frozen
    legacy code) has the identical `hasattr(layer, "lora_A")` gate, so
    it returns an EMPTY pair for a bare, top-of-stack DoRALinear/
    DoRAConv2d (composition, not inheritance -- lora_A/lora_B live at
    self._lora.lora_A) and never returns `magnitude` at all -- it
    predates DoRA, magnitude isn't a concept it knows to look for.

    This matters even with requires_grad correctly set:
    ComfyUNetTrainableModel.trainable_parameters() (nodes/model/
    lora_injector.py, this project's own code -- see it for where this
    gets combined with lora_parameters()'s own output) is what actually
    builds the list handed to an optimizer. An optimizer only ever
    steps on parameters explicitly given to it; requires_grad controls
    whether autograd computes a .grad for a tensor at all, not whether
    an optimizer that never received it will use one anyway. So
    reenable_dora_requires_grad() alone was necessary but not
    sufficient -- without this too, a real training run through
    ComfyUNetLoRANode(adapter_strategy=DoRAAdapter()) would have
    computed real gradients for every DoRA parameter (thanks to that
    fix) and then handed an optimizer a parameter list that silently
    doesn't include a single one of them, so none would ever actually
    move. Also fixes a second, smaller consequence of the same root
    gap: ComfyUNetTrainableModel.footprint_bytes() uses
    lora_parameters() (via data_ptr()) to decide which tensors are
    "the trainable adapter" versus "the frozen base" for memory
    reporting -- without this, every DoRA layer's lora_A/lora_B/
    magnitude would be miscounted as part of the frozen base's own
    footprint.

    Only a bare, top-of-stack DoRALinear/DoRAConv2d needs anything
    added here -- a DoRA layer wrapped in a later LoRAGeneration (this
    project's own nodes/model/lora_phases.py, after a phase split) has
    its OWN direct lora_A/lora_B that lora_parameters()'s hasattr check
    already matches correctly (LoRAGeneration doesn't use composition),
    and that frozen DoRA layer's own magnitude is correctly excluded
    here too -- split_into_new_generation() already explicitly froze it
    (requires_grad=False; see that function's own docstring for why),
    so it has no business being handed to a fresh phase's optimizer
    even if it were somehow reachable.
    """
    from .dora_layer import DoRAConv2d, DoRALinear

    params = []
    for _full_name, _parent, _attr, layer in registry:
        if isinstance(layer, (DoRALinear, DoRAConv2d)):
            A, B = layer.get_lora_weights()
            params.append(A)
            params.append(B)
            params.append(layer.magnitude)
    return params
