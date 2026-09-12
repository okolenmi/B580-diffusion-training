*[← docs/design index](README.md)*

# Precedent/validation, and implementation status

## 8. Note on precedent and validation

A few small things worth naming because they're evidence the design holds
together, not just assertions that it does: `LossWeighting` (section 4)
accepted a second implementation (`P2LossWeighting`) with zero interface
change. `Parameterization`'s `convert_to()` (1.4) already generalizes to
a third, velocity-target implementer without modification. `Algorithm`'s
`init_state()` (`nodes/optimizer/`) already permits a non-fp32,
non-plain-tensor state representation without needing a new method.
`AdapterStrategy` and `FrozenWeightStore` (3.1, 3.3) turned out to be
genuinely orthogonal axes, confirmed by QDoRA existing in the published
literature as the combination of both. None of these were required to
hold -- each is a place the design could have needed a revision it didn't
turn out to need.

**A stronger form of the same evidence exists now that isn't just
about interface stability under paper study: every piece in section 9.1
below actually got built, real-hardware-adjacent, equivalence-tested
against the exact behavior it replaced, and landed without needing a
design revision along the way.** That's a different, harder bar than
"the interfaces look right on paper" -- it's "the interfaces were right
when actual code had to satisfy them."

---

## 9. Implementation status: this design vs. current `nodes/`

### 9.1 What's implemented -- no further action needed

Everything below is real, tested code, not illustrative Python.
Everything the original section 9.1 table listed as "already matched
independently" is included here too, since the distinction between
"matched before this design started" and "built because of it" doesn't
matter anymore -- both are equally done.

| Design piece | Real `nodes/` location | Status |
|---|---|---|
| `Builder`/`Port` (1.1) | `nodes/core.py`'s `Node`/`Port` | Pre-existing, arrived at independently -- confirmed, not changed. |
| `Algorithm` x `ExecutionStrategy` x `Handle` composition (referenced throughout) | `nodes/optimizer/` in full | Pre-existing reference implementation -- this design's generalization target, not a gap. |
| `DeviceResident` (1.2) | `nodes/memory/handle.py`, conformed to by `OptimizerHandle`, `TrainableModel`, `TextEncoder` | Backlog items 3, 9, 12. |
| Pooled device buffers (1.3) | `nodes/memory/manager.py`'s `MemoryManager` | Pre-existing, unchanged interface; adoption breadth closed by the `DeviceResident` rollout above. |
| `NoiseSchedule`/`Parameterization`/`DiffusionProcess`/`DeviceContext` (1.4, 1.5) | `nodes/components/diffusion.py`, `nodes/components/device.py` | Backlog items 1-2. |
| `ProjectLayout` (1.6) | `nodes/components/layout.py` | Backlog item 8. Bridging period still open -- see 1.6. |
| `TrainingStepPipeline`/`StepPhase` (2.1) | `nodes/train/step_pipeline.py` | Backlog item 10. |
| `ActivationCheckpointingStrategy` (2.3) | `nodes/model/gradient_checkpointing.py` | Backlog item 7. |
| `BlockCost`/`CheckpointPlacementPolicy`/`EveryBlockPlacement`/`GreedyRatioPlacement` (2.3) | `nodes/model/checkpoint_placement.py` | Backlog item 1 (policy half). `GreedyRatioPlacement` unvalidated -- see 9.2. |
| `BlockProfileCollector`/`ProfilingCheckpointing` (2.3) | `nodes/model/block_profiler.py` | Backlog item 1 (instrumentation half -- the actual blocker). Only `ResBlock` instances ever reach it in this ComfyUI version -- see 2.3. Not wired into `ComfyUNetLoRANode`'s real construction path -- see 9.2. |
| Text encoder cache as `DeviceResident` (2.4) | `nodes/model/text_encoder.py`, `nodes/model/text_encoder_cache.py` | Landed as part of item 12. |
| `PrefetchingBatchSource` (2.5) | `nodes/dataset/prefetch.py` | Backlog item 11. |
| `AdapterStrategy`/`PlainLoRAAdapter`/`DoRAAdapter`, `LoRAScalingPolicy` (3.1, 3.2) | `nodes/model/adapter_strategy.py`, `nodes/model/dora_layer.py`, `nodes/model/lora_scaling.py` | Backlog item 9 (part 2), item 5, and formerly item 1. Live-wired into `ComfyUNetLoRANode`'s real construction path via `nodes/model/adapter_injection.py`'s `adapter_strategy_scope` -- see 3.1. `DoRAAdapter` grounded directly in HuggingFace PEFT's real source. Checkpoint save/load (direction + `.dora_scale` magnitude + alpha) now real for an unsplit DoRA layer -- see 3.1 and 9.2 for the one edge case still open. Two more real, previously-silent bugs in the same "composition, not inheritance" territory found and closed while landing the checkpoint work, both severe enough that a DoRA training run through `ComfyUNetLoRANode` trained nothing at all before either was fixed -- see `nodes/model/adapter_injection.py`'s `reenable_dora_requires_grad()`/`dora_trainable_parameters()`. |
| `adapter_strategy_scope` (3.1) | `nodes/model/adapter_injection.py` | Live-wires `AdapterStrategy` into `core.lora._inject_lora`'s real, unmodified targeting logic without modifying `core/lora.py`. See 3.1 for the mechanism and the recursion hazard it fixes. |
| `FrozenWeightStore`/`BF16WeightStore`/`NF4WeightStore` (3.3) | `nodes/model/frozen_weight_store.py`, `nodes/model/nf4_weight_store.py`, `nodes/model/nf4_lora_layer.py` | Backlog item 9 (part 1), and formerly item 1. Wired into a real forward path via `NF4LoRALinear`/`NF4LoRAConv2d` and a `frozen_weight_store` port on `ComfyUNetLoRANode` -- see 9.2 for the remaining real-run quality check. |
| `ParameterGroupPolicy`, `LoRAPlusGroups` (3.4) | `nodes/optimizer/composed.py` | Backlog item 4. `group_policy` port now exposed on every `Composed*OptimizerNode` (2.2) -- `LoRAPlusGroups` is real and selectable, but unvalidated -- see 9.2. |
| `ResourceBudget`/`ResourcePolicy`/`ManualResourcePolicy` (2.2) | `nodes/resource_policy.py` | Scoped to 3 of the design's original 7 methods -- see 2.2 for why. Wired into `ComfyUNetLoRANode` (`resource_policy` port) and, via a separate direct `group_policy` port, all three `Composed*OptimizerNode` classes. |
| `LossWeighting`/`LRSchedule` (section 4) | `nodes/train/loss.py`/`schedule.py` | Pre-existing clean ABCs, confirmed by `P2LossWeighting` needing zero interface change; v-pred branch + `P2LossWeighting` are backlog item 6. |
| `Algorithm.init_state()`'s state representation | `nodes/optimizer/algorithms/*.py` | Pre-existing -- contract already returns "a plain dict of named tensors," not specifically fp32; nothing structurally blocks a future quantized-state `Algorithm`. |
| `ResourceCoordinator`/`OffloadOrchestrator` (5.1, 5.2) | `nodes/memory/coordinator.py` | Backlog item 12. Doesn't by itself fix the still-open VRAM-hang report -- see 9.3. |
| `ResourceProfile` (5.5) | `nodes/memory/profile.py` | Backlog item 1. Wired into `SupervisedLoRATrainerNode`'s existing `profile=True` reporting (`nodes/train/step_pipeline.py`'s `MonitoringPhase`) -- `resident_<name>_mb` per-`DeviceResident` breakdown alongside the existing `tracked_footprint_mb` total, same gate, no new port. `memory_manager_stats` is `None` in a real run today -- real gap found while landing this, not fixed here: no shared `MemoryManager` instance is reachable from `SupervisedLoRATrainerNode.build()` to pass to `capture()`; `ChunkedScratchBufferStrategy` (`nodes/optimizer/strategies/chunked.py`) constructs its own private one when none is injected, and nothing upstream injects a shared one. See `nodes/memory/profile.py`'s module docstring. |
| Injected pub/sub, not a singleton bus (5.2's reasoning) | `nodes/monitor/`'s `MonitorHandle`/`LiveMonitorHandle` | Pre-existing house reference for "no singleton" done right -- explicitly reused, not redesigned, for `OffloadOrchestrator`. |
| Decorator-wrapped `TrainingBatchSource` (2.5) | `nodes/dataset/renoise.py`'s `RenoiseBatchSource` | Pre-existing pattern `PrefetchingBatchSource` followed. |
| Composition-over-mutation for stacked state (1.2) | `nodes/model/lora_phases.py`'s `LoRAGeneration` | Pre-existing, independent instance of the same principle `DeviceResident`'s offload-vs-free distinction is built on. |
| `server/graph_executor.py` | -- | Pre-existing, already matches this design's construction-time model closely: real topological execution, real `issubclass()`-based port compatibility checking, explicit `ExecutionContext` threading. No changes recommended. |

### 9.2 What's still missing, partial, or unvalidated

Everything below is real -- these are the only pieces of this document
still asking for something. See section 10 for the ordered plan.

**`NF4WeightStore`'s real-run quality check (3.3).** The forward path is
wired now -- `NF4LoRALinear`/`NF4LoRAConv2d`
(`nodes/model/nf4_lora_layer.py`), equivalence-tested against a fixed
dequantized reference, plus a `frozen_weight_store` port on
`ComfyUNetLoRANode`. What's missing: verification against this
project's own real UNet, not assumed from the LLM literature (real
~9% relative RMSE quantization error, does it still train usably), and
a `MemoryManager`-backed scratch buffer for the dequantized tensor
(real, separate VRAM optimization, not a correctness blocker -- a fresh
allocation per forward call is already correct, just not maximally
efficient).

**Four real gaps found and closed across two sessions of wiring in
DoRA's checkpoint round-trip, one real gap narrowed and left honestly
open (3.1).** Every one of the first three shares the same root cause:
`DoRALinear`/`DoRAConv2d` (`dora_layer.py`) hold their `lora_A`/`lora_B`
nested one level down (`self._lora.lora_A`), built via composition, not
inheritance -- so any code elsewhere that assumes a LoRA layer's
direction lives as its own direct attribute, rather than going through
the `get_lora_weights()` contract every layer kind actually implements,
silently mishandles a DoRA layer specifically. Each instance below was
found independently, by actually exercising the real path, not by
auditing for this pattern in advance -- the pattern only became visible
once enough of them had turned up to name it.

The checkpoint gap itself is closed for the common case: `.dora_scale`
(direction, magnitude, and alpha) now round-trips exactly through
`LoRACheckpointSaverNode`/`LoRACheckpointLoaderNode` for a DoRA layer
that's never been phase-split -- see 3.1 and 9.1's table.

Found in the process, and *not* the gap that was being looked for:
`nodes/model/lora_phases.py`'s `split_into_new_generation` (the function
`LoRAPhaseSplitNode` calls) reached for `layer.lora_A`/`layer.lora_B` as
direct attributes when freezing the previous generation, so
phase-splitting a DoRA layer raised a plain `AttributeError` the instant
anyone actually wired `DoRAAdapter` into `LoRAPhaseSplitNode` -- a
combination the graph editor's own type contracts have accepted as
legal this whole time, with nothing about it hinting the combination had
never actually been exercised. Fixed by freezing through
`get_lora_weights()` instead of assuming the attribute layout
underneath -- `LinearLoRAGeneration`/`Conv2dLoRAGeneration._build_params()`
had the identical assumption one level up and needed the same fix. A
second, genuinely silent issue the same fix would have missed on its
own: `get_lora_weights()` returns direction only, so `magnitude` needed
freezing explicitly too, or a fresh phase-2 optimizer would have kept a
"frozen" phase's magnitude receiving real gradient updates for as long
as phase 2 trained. Both fixed; verified with an actual gradient-
isolation check (magnitude provably untouched, bit-for-bit, after
training the new generation), not just that it no longer crashes.

Found in a second pass, deliberately looking for other instances of the
same pattern, and by far the most severe of the four: **a real DoRA
training run through `ComfyUNetLoRANode(adapter_strategy=DoRAAdapter())`
trained nothing at all**, in two independent, both-necessary ways.
`core.unet_wrapper.ComfyUNetWrapper._init_lora()` (frozen legacy code)
freezes every model parameter, then re-enables `requires_grad` only for
whatever passes `hasattr(layer, "lora_A")` -- False for a bare DoRA
layer, so `lora_A`, `lora_B`, and `magnitude` (which that function
doesn't even know exists -- it predates DoRA) all stayed frozen, with no
error and nothing printed. `nodes/model/adapter_injection.py`'s new
`reenable_dora_requires_grad()` fixes this from outside `core/`, the
same way `adapter_strategy_scope` already works around a different
`core/` assumption. That alone would still not have been enough:
`ComfyUNetWrapper.lora_parameters()` -- what
`ComfyUNetTrainableModel.trainable_parameters()` actually hands to the
optimizer -- has the identical `hasattr` gate, so even with
`requires_grad` correctly restored, the optimizer built from it would
never have received a single DoRA parameter to step on (`requires_grad`
governs whether autograd computes a gradient at all, not whether an
optimizer that was never given a parameter updates it anyway). The new
`dora_trainable_parameters()`, combined into
`ComfyUNetTrainableModel.trainable_parameters()`, closes this side too
-- and the same combination fixes a smaller side effect of the identical
gap in `footprint_bytes()`, which had been silently counting DoRA's own
trainable tensors as part of the frozen base. Verified with a real,
two-step optimizer loop that provably moves every DoRA parameter (not
zero-init masking a still-broken gradient path), not just that
`requires_grad` reads `True`.

What's still real and honestly left open, narrower than before: a
phase-split DoRA layer's magnitude still can't be folded into a
*combined*, multi-generation checkpoint. `extract_combined_weights` now
raises a clear, explanatory error for this case instead of silently
emitting a checkpoint that quietly drops the trained magnitude's effect
-- magnitude scales the *entire* frozen-base-plus-delta result, not
expressible as "one more rank-stacked generation" the way a plain
generation's own delta is. `extract_own_generation_weights` (the "just
this phase" snapshot `LoRAPhaseSplitNode.completed_generation` actually
uses) has no such limitation and round-trips a DoRA phase's magnitude
fine either way, since it never combines. How phase-splitting and a DoRA
base's magnitude should even combine, if at all, is a real, separate
design question -- not attempted here.

**`GreedyRatioPlacement` real validation (2.3).** The class, and the
`BlockProfileCollector`/`ProfilingCheckpointing` instrumentation that
was actually blocking it, are both implemented and equivalence-tested
now -- see 9.1. What's missing is the same kind of thing missing from
`RescaledZeroTerminalSNRSchedule` below: a real profiled run producing
real `BlockCost` numbers for this project's actual UNet, then a real
placement decision made from them and compared against
`EveryBlockPlacement`'s baseline. Not wired into `ComfyUNetLoRANode` yet
either way -- `use_checkpoint=True` still means `EveryBlockPlacement`'s
unconditional behavior, real or not.

**`RescaledZeroTerminalSNRSchedule` real end-to-end validation (1.4).**
The class itself is implemented and wired (unlike everything else in
this list) -- what's missing is a real training run with
`VPredParameterization` and qualitative image-quality evaluation. Unlike
every closed item, there's no old code path to equivalence-test against;
this needs real training runs to trust, not a unit test.

**`LoRAPlusGroups` real tuning (3.4).** The class is implemented and, as
of 2.2's `group_policy` port, actually selectable from the graph on every
`Composed*OptimizerNode` (unlike everything else in this list) -- what's
missing is actually running a LoRA training job with it wired in and
comparing against a `UniformGroups` baseline, at whatever `ratio` turns
out to matter for this project's own data.

**`ComponentRegistry`/`TrainingRecipe`/`PipelineFactory` (5.3, 5.4).**
None exist. Still not recommended as near-term work (see 5.3, 5.4 for
the current reasoning -- `nodes/components/` now has real content, but
nothing in it is graph-editor-selectable, so the side-by-side-
registration problem these solve still hasn't materialized).
`ResourceProfile` (5.5), the fourth item this paragraph used to list, is
done -- see 9.1.

**A live, per-step VRAM budget enforcer, new since the list above --
genuinely different from `OffloadOrchestrator` (5.2), not a rename of
it.** `OffloadOrchestrator` is still exactly as un-wired as this
document already said: event-driven, for three specific, rare moments
(cache rebuild, preview generation, checkpoint save), and nothing in
the real training loop publishes those events yet. `ResourceControlHandle`/
`BudgetedResourceControlHandle` (`nodes/memory/control_handle.py`) is a
different, complementary shape for a different problem: not "react to
a named, rare event," but "check real measured usage before every
single step, offload whatever's marked safe to if over a stated
`ResourceBudget` (5.5's own type, reused as-is), reload it right before
whatever needs it next actually needs it." Built as a handle one node
constructs (`VRAMBudgetControllerNode`) and another (the trainer) calls
into during its own `build()`, the same shape `MonitorHandle`/
`LiveMonitorHandle` already established for a different cross-cutting
concern -- deliberately not a second graph node running "alongside" the
trainer, since `server/graph_executor.py` runs nodes in topological
order, one at a time; there's no mechanism for two nodes to run
concurrently and exchange live signals mid-execution, so a callback
object one node hands to another is what makes "continuous" possible at
all here. Wired into `SupervisedLoRATrainerNode` (a new
`resource_control` input) -- real measurement happens every step when
one is connected. Honestly incomplete in one specific way: nothing
registered with it (`model`/`optimizer`/`text_encoder`) is currently
marked offloadable by default, because none of them has a genuine idle
window in this pipeline's own current, always-synchronous design (text
encoding, for instance, runs unconditionally every step -- see
`EncodeConditioningPhase`, section 4). This was groundwork with a real,
tested mechanism underneath it more than a today-provides-relief
feature -- true for one revision, resolved the next: offloading
something for real needs that something to have an actual idle window
first, and `CachingTextEncoder` (`nodes/model/text_encoder_cache.py`,
already existed, predating this specific addition) is exactly that --
an LRU cache in front of any `TextEncoder`, skipping the inner
model entirely on a hit. Wired together directly: `CachingTextEncoder`
takes an optional `resource_control`; on a cache miss it calls
`ensure_loaded()` before falling through to the inner encoder, so it's
always safe to have been offloaded between hits. `ensure_loaded()`
itself grew a second responsibility to make this actually safe under
real pressure, not just a happy-path reload: it now shares a
`_make_room()` helper with `before_step()`, so reloading one resident
that would push measured usage over budget offloads *other* offloadable
residents first to make room -- direct feedback describing the exact
case this needs to handle (model/optimizer already near budget when a
cache miss needs the text encoder loaded). `SupervisedLoRATrainerNode`
marks `text_encoder` offloadable exactly when it's actually a
`CachingTextEncoder` (checked via `isinstance`, matching this file's
own existing `FusedOptimizerHandle` check, not assumed) -- a plain,
non-caching encoder has no such self-healing and stays
`offloadable=False`, unchanged. `model`/`optimizer` also stay
`offloadable=False` still -- both are needed unconditionally every
step's compute, and nothing yet calls `ensure_loaded("model")`/
`ensure_loaded("optimizer")` at the right point in the step pipeline to
make offloading either of them safe. `_make_room()` would already
handle that side of a swap correctly if it existed; the wiring to
trigger it doesn't yet.

### 9.3 What's explicitly out of scope

`core/trainer.py` and the rest of `core/`/`manager/` are the production
path, reference material only, untouched by this design -- exactly the
existing project rule. The VRAM-pressure hang/device-lost report in
`docs/suspicious_findings.md` lives there today; this design's
`OffloadOrchestrator` is the *eventual*, principled home for that class
of coordination problem once `nodes/` is the production path, not a
claim that building it retroactively fixes `core/trainer.py`'s current
hand-rolled offload logic.

---
