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
about interface stability under paper study: every piece in this design
actually got built, real-hardware-adjacent, equivalence-tested against
the exact behavior it replaced, and landed without needing a design
revision along the way.** That's a different, harder bar than "the
interfaces look right on paper" -- it's "the interfaces were right
when actual code had to satisfy them."

---

## 9. Implementation status: this design vs. current `nodes/`

### 9.1 What's implemented -- no further action needed

Deliberately one line: what's implemented is the source tree, so read
`nodes/` rather than a table re-typing it. Section 9.2 below is the part
still worth reading.

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
that's never been phase-split -- see 3.1.

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
`unet_wrapper.ComfyUNetWrapper._init_lora()`
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

### 9.3 What's explicitly out of scope

`core/trainer.py` and the rest of `core/`/`manager/` are the production
path, reference material only, untouched by this design -- exactly the
existing project rule. The VRAM-pressure hang/device-lost report in
`docs/known-issues/open.md` lives there today; this design's
`OffloadOrchestrator` is the *eventual*, principled home for that class
of coordination problem once `nodes/` is the production path, not a
claim that building it retroactively fixes `core/trainer.py`'s current
hand-rolled offload logic.

---
