"""adapter_strategy_scope: makes AdapterStrategy (adapter_strategy.py)
the real thing ComfyUNetLoRANode builds, not just an equivalence-tested
seam sitting next to the actual construction path.
See docs/training_pipeline_design.md section 3.1/10 for the rationale.

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

**Why PlainLoRAAdapter installs no patch at all.** PlainLoRAAdapter.wrap()
(adapter_strategy.py) itself constructs LoRALinear/LoRAConv2d by
importing them fresh from core.lora inside its own wrap() body -- if
this module patched those same names and then routed PlainLoRAAdapter
through the patched path, PlainLoRAAdapter.wrap()'s own import would
resolve to the patch it's currently running inside of, recursing
forever. There's a real, structural reason this never comes up:
PlainLoRAAdapter *is* core.lora's own unmodified behavior by
definition, so there is nothing to intercept when it's selected --
skipping the patch entirely for this one case is not an optimization
shortcut, it's the only correct behavior, and it happens to also be the
default (adapter_strategy=None -> PlainLoRAAdapter()), so nothing
wired to ComfyUNetLoRANode today changes at all.

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


def _make_adapter_patched_class(adapter_strategy: AdapterStrategy):
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
    already deciding correctly."""

    class _AdapterPatchedLayer:
        def __new__(cls, original, rank: int = 64, alpha: float = 1.0,
                    dropout: float = 0.0, weight: float = 1.0):
            frozen = BF16WeightStore(original.weight)
            # ClassicLoRAScaling(), always -- see this module's docstring
            # for exactly why the real scaling_policy must not be passed
            # here.
            return adapter_strategy.wrap(original, frozen, rank, alpha,
                                          ClassicLoRAScaling(), dropout, weight)

    return _AdapterPatchedLayer


@contextlib.contextmanager
def adapter_strategy_scope(adapter_strategy: AdapterStrategy):
    """Everything core.lora._inject_lora constructs inside this `with`
    block goes through adapter_strategy instead of core.lora's own
    LoRALinear/LoRAConv2d -- unless adapter_strategy is a
    PlainLoRAAdapter, in which case nothing is patched at all (see this
    module's docstring for why that's correct, not a shortcut).
    Restores core.lora's real classes on every exit, exception or not.
    """
    if isinstance(adapter_strategy, PlainLoRAAdapter):
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
    patched = _make_adapter_patched_class(adapter_strategy)
    core_lora.LoRALinear = patched
    core_lora.LoRAConv2d = patched
    try:
        yield
    finally:
        core_lora.LoRALinear = original_linear
        core_lora.LoRAConv2d = original_conv2d
