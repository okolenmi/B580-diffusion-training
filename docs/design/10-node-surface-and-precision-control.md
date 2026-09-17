*[← docs/design index](README.md)*

# 11. Node surface and precision control (planning)

Real, user-reported ground for this section, not a self-directed
exercise: real-hardware testing found individual optimizer nodes
(`AdafactorOptimizerNode`, `CAMEOptimizerNode`, etc.) performing within
~10% of their `Composed*` equivalents -- "legacy LoRA handling should be
replaced with new one for sure" -- plus no coherent way to control
compute/storage/optimizer-state precision, plus (found live, mid-session,
the hard way) a real duplication bug: `strategy="shape_grouped"` was
registered on `ComposedCAMEOptimizerNode` but not on
`ComposedAdafactorOptimizerNode`/`ComposedAdamWOptimizerNode`, because
each of the three composed nodes carried its own byte-identical copy of
the same dict, dispatch logic, and doc string. The first fix (registering
the missing entry in the two other copies) reintroduced the identical bug
class in the process -- the doc strings went stale in exactly the same
way. Every proposal below is written with that lesson applied on purpose:
shared structure first, no per-node/per-algorithm copies of the same
thing -- see `nodes/optimizer/strategy_registry.py` for the real fix, and
`docs/known-issues/resolved.md`'s matching entry for the full account.
11.4/11.5 add two further real, user-reported points from the same
conversation: `strategy`/`device`-style ports being free-text strings
with no discoverable set of choices, and every node's Python class name
also doubling as its palette display text, "...Node" suffix included.

## 11.1 Optimizer node consolidation

**Update, 2026-09-17: fully executed, not just planned any more.**
What follows is the current, actual state, not the original plan --
kept in one place rather than split into "the plan" and "what actually
happened" as two separate write-ups. (Tracked during development in
docs/CLEANUP_TODO.md, since deleted once nothing was left in it --
see git log for the day-by-day account, including two real corrections
made along the way that this section's original version got wrong.)

Grounded in actually reading every node file, not assumed from naming
alone -- the picture is real but uneven, not a blanket "delete the old
ones":

**Retired, confirmed equivalent:** `AdamWOptimizerNode` (wrapped
`CPUAdamW`), `CAMEOptimizerNode`, `ForeachCAMEOptimizerNode`,
`ForeachAdafactorOptimizerNode`, `FusedAdafactorOptimizerNode` -- each
was a thin pass-through wrapper around a legacy `core.optimizers`
class, and the matching `Composed*OptimizerNode` + `strategy=` choice
covers the same ground. CAME/ForeachCAME were already
equivalence-tested in `nodes/smoke_tests/` before this pass;
Foreach-Adafactor and Fused-Adafactor each needed a new, dedicated
real-torch run first (see `smoke_test_adafactor_tiny_parameter_gap.py`
and `smoke_test_fused_adafactor_equivalence.py`) -- this section's own
original version grouped Foreach with `AdafactorOptimizerNode` below as
though they shared one mechanism, which running the actual numbers
disproved (see next entry), and treated Fused's gap as unclosable
algorithm work rather than the small, opt-in change
(`AdafactorAlgorithm.tiny_parameter_threshold`) it turned out to be.
`AdamWOptimizerNode` retired for a different reason than "proven
numerically equivalent": `CPUAdamW`'s CPU-resident state solves a
full-fine-tune-parameter-count problem this project has no way to
produce at all (see `nodes/optimizer/composed_adamw.py`) -- the design
point this section's original version called "genuinely different...
stays regardless" turned out to have no actual consumer anywhere in the
codebase to be different *for*.

**Not retired, a real gap, confirmed on an actual torch run (not just
static reading):** `AdafactorOptimizerNode` alone now -- differs from
`ComposedAdafactorOptimizerNode` for small (< 10,000 element)
parameters, for a reason this section's original version got wrong (a
single, uniform `TINY_NUMEL` special case shared with the other two
Adafactor variants -- it wasn't shared, see the "Retired" entry above).
`ChunkedXPUAdafactor` batches every tiny parameter across the whole
optimizer into one shared clip/EMA state -- a cross-parameter,
execution-strategy-level concern, not a per-parameter algorithm one
(unlike `FusedXPUAdafactor`'s tiny-parameter mechanism, which *was*
per-parameter and got closed the same way the rest of this list did);
closing it needs new `ExecutionStrategy` machinery (something like
`ShapeGroupedBatchStrategy`, but grouping "under a size threshold"
instead of "same shape"), not an `AdafactorAlgorithm` change -- real,
separate, sized future work, tracked in
`docs/design/09-prioritized-backlog.md`, not something this pass
attempted.

**A real, two-directional capability difference, not one-directional
redundancy -- flagged, not yet acted on:** `SimpleAdamWOptimizerNode`
(`torch.optim.AdamW`, PyTorch's own first-party kernel, no custom code
at all) vs. `ComposedAdamWOptimizerNode` (this project's own
`AdamWAlgorithm`, which supports `group_policy=LoRAPlusGroups(...)` and
`strategy="shape_grouped"`, neither of which `SimpleAdamWOptimizerNode`
can do). This section's original argument for keeping both was: one is
"trust PyTorch's own AdamW, nothing fancier, minimal maintenance
surface," the other is "AdamW plus this project's own composable
pieces" -- a legitimate, real distinction distinct from
`AdamWOptimizerNode`'s case above (which had no real consumer;
"minimal trust surface" has one, in principle, regardless of consumer
count). **`SimpleAdamWOptimizerNode` was deleted anyway, in this same
pass** (`SimpleAdamWOptimizerNode` deleted in the first commit of this
pass, before this section was found) -- decided at the time
without weighing this specific argument (this section wasn't found
until several commits later). Worth a maintainer decision, not
silently re-added or silently left deleted: this project's established
direction is its own verified implementation over wrapping (see
`docs/architecture.md`), which argues for staying deleted -- but "trust
the standard library's own kernel for something this fundamental" is a
real, different kind of argument than the ones already weighed
elsewhere in this cleanup, and deserves an explicit answer rather than
inheriting one by default.

**Net-new, nothing to consolidate:** `ComposedFusedAdamWOptimizerNode`/
`ComposedFusedCAMEOptimizerNode` have no legacy equivalent at all.

**Deprecation approach -- changed from the original plan, on explicit
direction:** the original version of this section argued for marking
retired nodes as legacy in their own docstrings rather than deleting
the classes outright, specifically because "real graphs may already
reference them by name, and removing a registered `Node` class is a
one-way door for anyone's saved graph." That's a real concern in
general, but not the direction actually taken here: the four "retired"
nodes above were deleted outright, not soft-deprecated, on the project
owner's explicit instruction to converge on one concrete implementation
per concern rather than carry parallel options indefinitely, with git
history (not a deprecated-but-present class) as the safety net for
anyone who needs the old behavior back. Recorded here so the tradeoff
this section originally weighed, and the different call actually made,
are both visible -- not just the outcome.

## 11.2 `ExecutionStrategy` is up to three orthogonal axes, not one flat enum

Directly answers a real question raised this session -- "why is
`shape_grouped` a completely different strategy if in theory it can be
combined with other things" -- by actually reading every strategy's
`step()`, not by guessing at the answer:

1. **Is the per-parameter algorithm math batched across same-shape
   parameters?** None (`SimpleLoopStrategy`/`ChunkedScratchBufferStrategy`/
   `ForeachApplyStrategy` all call `Algorithm.compute_update()` in a
   plain per-parameter loop) vs. `ShapeGroupedBatchStrategy`'s
   `compute_update_batched()`, one call per same-shape group.
2. **Is the final apply step (`decay`/`delta` onto `param.data`)
   batched?** A per-parameter Python loop (`apply_update()`, used by
   `SimpleLoopStrategy` and `ShapeGroupedBatchStrategy` alike -- confirmed
   directly: `ShapeGroupedBatchStrategy` batches the *math* but still
   applies each group member's result one at a time) vs.
   `ForeachApplyStrategy`'s `torch._foreach_*` calls across `(device,
   dtype)` groups.
3. **Does `compute_update()` get a `MemoryManager`-backed scratch buffer
   for its own internal intermediates?** None (fresh allocation every
   call) vs. `ChunkedScratchBufferStrategy`'s reused buffer.

Today's four strategies explore three of the many combinations:
`simple` = (none, none, none), `chunked` = (none, none, scratch),
`foreach` = (none, foreach, none), `shape_grouped` = (math, none, none).
**The valuable, unexplored combination was (math, foreach, none)** --
batching the core math *and* the apply step compounds the two
strategies' separate overhead reductions.

**Implemented**: `ShapeGroupedForeachStrategy`
(`nodes/optimizer/strategies/shape_grouped_foreach.py`) -- built from
two already-proven pieces, not re-derived: the grouping/batched-compute
logic (`compute_update_batched()`) was extracted out of
`ShapeGroupedBatchStrategy` itself into `strategies/shape_grouping.py`
once this strategy needed the identical logic too, and the batched-apply
logic (including the bf16 rounding fix) was extracted out of
`ForeachApplyStrategy` into `base.py`'s `apply_updates_batched()` the
same way. Both original strategies were refactored to call the
extracted, shared versions rather than keeping their own copies --
applying section 11.0's lesson to this section's own proposal, not just
citing it. **Not** a general N-axis composable-strategy framework for
two boolean-ish axes and one missing combination -- that would have been
over-engineering ahead of actual need, against this project's own "don't
overcomplicate" rule (section 0). If a third or fourth genuinely
independent axis shows up later, that's the trigger to actually
decompose `ExecutionStrategy` into composable pieces, not before.

**A real, latent correctness bug found while building and testing this,
not specific to the new strategy:** `Algorithm.compute_update_batched()`'s
default fallback (`algorithms/base.py`, used by any Algorithm without
its own batched override, and by `AdafactorAlgorithm.compute_update_batched()`'s
own `scale_parameter=True` fallback) silently kept only the *last* group
member's `decay` when `decay` genuinely varied across the group --
exactly `AdafactorAlgorithm`'s `scale_parameter=True` case, since
`alpha_t` depends on each parameter's own norm. An earlier equivalence
test of this exact fallback path happened to use `weight_decay=0.0`,
where `decay` is always `None` regardless of `alpha_t` -- silently
avoiding the bug rather than proving its absence. Found by a real
equivalence-test failure once `weight_decay != 0` was actually exercised
through a batched strategy, not by inspection. Fixed to raise a clear,
specific `RuntimeError` instead of silently applying the wrong decay --
see `docs/known-issues/resolved.md`'s matching entry.

**Fused execution is a fourth thing, but not a fourth axis of
`ExecutionStrategy` at all.** `ComposedFusedOptimizerHandle`
(`composed_fused.py`) applies each parameter's update the instant *its
own* gradient is ready, from inside a backward hook, before `backward()`
has even returned -- confirmed directly from that module's own
docstring, which explains exactly why this couldn't be "one more
strategy" (`step()` is never meaningfully called at all). Combining
shape-grouped/foreach batching with fused execution is a genuinely
harder problem, flagged honestly rather than glossed over: batching
needs to *wait* for a whole group's gradients to arrive; fused wants to
act on each parameter the moment it's ready, which is in tension with
waiting for anything. Not proposed here -- a real, separate, harder
question if it's ever worth pursuing.

## 11.3 Precision and storage control

Three currently real, currently under-exposed, and currently *separate*
dtype decisions -- kept as independent choices rather than one bundled
"precision mode," matching section 2.2's own precedent
(`adapter_strategy`/`frozen_weight_store` were deliberately kept out of
`ResourcePolicy` rather than absorbed into one object, back when
`ResourcePolicy` still existed -- see 2.2 for why it was removed since):

1. **`frozen_weight_store`** -- implemented: a real port on
   `ComfyUNetLoRANode`, mirroring `adapter_strategy`'s own pattern
   (default `None` -> `BF16WeightStore`, `NF4WeightStore` selectable,
   works with the default `PlainLoRAAdapter`). A real gap found and
   fixed while wiring this: `adapter_strategy_scope`'s "skip patching
   for `PlainLoRAAdapter`" shortcut assumed that always meant bf16 --
   wrong once `PlainLoRAAdapter` could also mean NF4 storage, which
   would have silently done nothing under the old rule. The skip
   condition now checks both `PlainLoRAAdapter` *and* the `BF16WeightStore`
   factory, not `PlainLoRAAdapter` alone.
2. **`state_dtype`, now done -- as block-wise 8-bit quantization, not
   the plain dtype cast this item originally described.** Every
   `Algorithm.init_state(param_shape, dtype, device)` still ignores
   its own `dtype` argument, hardcoding float32 internally -- unchanged,
   still correct for the same numerical-stability reason (a raw bf16
   cast of momentum/variance was and still is a real, unvalidated
   numerical risk, not "does it run," exactly as this item originally
   flagged). What shipped instead answers the same underlying need
   (configurable optimizer-state memory) a different, better-validated
   way: `OptimizerStateStore`/`Int8BlockStateStore`
   (`nodes/optimizer/state_store.py`) block-wise quantizes `m`/`v` to
   8 bits between steps (dequantize to real fp32 -> `Algorithm.
   compute_update()` runs completely unchanged, unaware this exists ->
   requantize the result), ~4x smaller than the bf16 idea's 2x, and
   verified end to end: a real 20-step AdamW training comparison
   (`Float32StateStore` vs `Int8BlockStateStore`, same seed, same
   gradients) converges to within 0.0025 max per-parameter difference,
   not just "produces finite numbers." **One shared implementation**
   (11.0/11.1's own lesson, applied here too): `state_precision`'s
   choices/doc/resolver live once in `state_store.py`, the same
   `STRATEGIES`/`resolve_strategy()` shape `strategy_registry.py`
   already established for a different Port on the same three nodes.
   The earlier "Update" below (superseded, kept only so this entry's
   own history stays visible) guessed this would end up inside the
   Resources Controller redesign's own precision handling -- it didn't:
   that redesign's own Phase 5 settled on `ResourcesControllerNode`
   never touching LoRA injection or optimizer construction at all (see
   `docs/design/resources-controller/05-phase-5-resources-controller-node.md`), so
   optimizer state precision stayed exactly where `strategy`/`device`
   already lived, on the `Composed*` optimizer nodes themselves --
   consistent with, not a special case of, everything else on those
   nodes. ~~**Update:** likely belongs inside the Resources Controller
   redesign's own precision handling rather than as an isolated port on
   each optimizer node -- see
   `docs/design/resources-controller/08-consolidation.md` for the
   reasoning. Not decided; flagged
   there so this doesn't get implemented in isolation before that's
   settled.~~
3. **compute dtype** -- already real (`ComfyUNetLoRANode.dtype`), no new
   work needed, just clearer documentation that this *is* the
   compute-dtype axis. True autocast-based mixed precision (fp32 master
   weights, bf16 compute) is a separate, larger, not-yet-designed item --
   explicitly not claimed as covered by anything above.

**No preset bundle proposed** (e.g. one `"qlora"` flag setting several
of the above at once) -- matches the same orthogonality precedent
`ResourcePolicy` set while it existed (2.2): stay orthogonal rather than
pre-bundling choices nobody's asked to have bundled. Worth adding later
if real, explicit demand shows up, not speculatively now.

## 11.4 Port UX: string fields for closed-choice values

**Implemented**, as part of
`docs/design/resources-controller/08-consolidation.md`, exactly as this
section's own "Update" below
anticipated -- `Port.choices` (`nodes/core.py`), not per-node. See that
plan for what shipped, what it's wired into today (`strategy` on the
`Composed*OptimizerNode` classes, `t_mode` on the two real dataset
nodes), and why `device` deliberately stays open-ended rather than
getting a closed list. The original planning text below is kept as the
rationale for *why*, per this document's own top-of-file rule for
implemented sections.

A real, valid complaint from this session, not specific to optimizers:
`strategy`, `device`, and similar `Port`s are typed as bare `str`, with
the valid choices only discoverable by reading a doc string, guessing,
or hitting a runtime `ValueError`. This is a `Port`/graph-editor-level
gap, not a per-node one -- fixing it for `strategy` alone would just be
another single-node patch of a structural problem, the exact mistake
section 11.0 is about. A `Port` needs a way to declare a closed set of
valid choices (e.g. an optional `choices: list[str] | None` field,
`None` for genuinely open-ended strings) that server/graph-editor code
could render as a dropdown and validate before a graph even runs, not
just at `build()` time. Real, worth doing, but touches `core.py`'s
`Port` dataclass and the server's node-introspection/UI code -- a larger
item than anything else in this section, flagged honestly as its own
piece of work, not bundled into the optimizer-specific items above.

**Update:** this and the Resources Controller redesign's
`ResourcePreset` "parameter-value dictionary" are the same mechanism
underneath the different names -- a `Port` with a closed set of
choices, rendered as a dropdown, resolved to one value by build time.
Recommend building this once as part of that redesign's Phase 3/4
rather than as a separate, disconnected item -- see
`docs/design/resources-controller/08-consolidation.md`.

## 11.5 Node naming: one string is doing two jobs

**Implemented.** `Node.DISPLAY_NAME` (`nodes/core.py`, default `None`) and
`NodeInfo.display_name` (`server/nodegraph_introspect.py`, computed by the
new `_auto_display_name()`/`_display_name_for()`) exist and are wired into
`/nodegraph/registry`'s response and the palette (`server/static/nodegraph.js`),
exactly matching the proposal below -- `class_name` is untouched, every
saved graph keeps resolving the same way it always did. One real
discrepancy between the proposal and what actually shipped, found while
implementing it, not glossed over: the proposal's own worked example
(`"ComfyUNetLoRANode"` -> `"Comfy UNet LoRA"`) is **not** achievable by
literally "split on capitals" as written below -- a plain capital-boundary
split turns that same input into `"Comfy U Net Lo R A"`, since `"UNet"`
and `"LoRA"` are each two capitals-then-lowercase runs, not one. What's
actually implemented is a curated, closed list of this project's own
domain tokens (`UNet`, `LoRA`, `DoRA`, `CAME`, `AdamW`, `SDXL`, `SNR`, `NF4`,
`BF16`, `XPU`, `VRAM`, `LR`, `P2`, ...), checked longest-match-first at each
position before falling back to a generic capital-then-lowercase word --
see `_auto_display_name()`'s own docstring in `nodegraph_introspect.py`.
Verified against every one of the 36 classes actually in
`nodegraph_registry.get_registry()` today, not just the doc's one example
(`server/smoke_tests/smoke_test_nodegraph_introspect.py`), plus the
`DISPLAY_NAME` override path itself, via synthetic classes so the override
check doesn't depend on any real node happening to set one yet -- none do.

The original planning text below is kept as the rationale for *why*, per
this document's own top-of-file rule for implemented sections; the
illustrative naming detail (checked and refined during landing, per the
paragraph above) says exactly what shipped now, not what was speculated
before.

A second real UX point, raised alongside 11.4: every node's Python class
name (`ComfyUNetLoRANode`, `ComposedAdafactorOptimizerNode`, ...) is used
as *both* the stable identifier a saved graph's `class_name` references
and resolves against (`server/nodegraph_registry.py`'s `get_registry()`
returns `{cls.__name__: cls for cls in classes}`, one dict, deliberately
"so those two things can never silently disagree" -- but that's *only*
true because there's just the one string) *and* what a person sees in
the palette -- including the "...Node" suffix on every single entry,
which carries no information (everything in the palette is a node) and
reads exactly as "dumb" as it was called.

These two jobs have different, real constraints: the registry key needs
to be stable (renaming it breaks every already-saved graph referencing
the old name) and needs to be a valid Python identifier; a display label
has neither constraint and should read well to a person scanning a
palette. Collapsing them into one string means neither job is served
well -- `class_name` can't be prettied up without breaking saved graphs,
and there's no way to *ever* improve the palette's wording without that
same breakage.

**Concrete, low-risk proposal, matching this module's own stated
principle** (`server/nodegraph_introspect.py`: "UI metadata is *derived*
from the real Python class... never hand-duplicated in a separate
file"): add an optional `display_name` to `NodeInfo`
(`nodegraph_introspect.py`), derived automatically (strip a trailing
`"Node"` suffix, split on capitals -- `ComfyUNetLoRANode` ->
`"Comfy UNet LoRA"`) with a class-level override available (e.g. a
`DISPLAY_NAME: ClassVar[str] | None` on `Node` itself, `core.py`) for the
cases an auto-derived name reads badly. `class_name` stays exactly what
it is today -- the registry key, `__name__`, untouched, so nothing about
serialization or graph resolution changes at all. Purely additive: every
existing saved graph keeps working unchanged; only the palette's own
rendering gets a second, friendlier string to show instead of the raw
class name.
