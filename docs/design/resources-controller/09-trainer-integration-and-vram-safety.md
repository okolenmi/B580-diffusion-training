*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 9 -- the Resources Controller route's own trainer, built around residency instead of gated onto the main route's

**Goal:** unblock the one thing Phase 6 left open (`LoRATrainingConfigNode`'s
`trainer` output had nowhere to plug in), and build the Resources
Controller route as a genuine second, memory-optimized alternative to
the existing main route -- with its own training step loop, not the
main route's loop reused behind a stricter gate. See "First attempt,
reverted" below for why that distinction is the whole point of this
phase, not a style preference.

**Status: done, verified against a focused smoke-test run (not the
full suite -- see "Verified" below for which tests and why only
those), not yet run on real hardware.**

### First attempt, reverted

The first version of this phase extracted `SupervisedLoRATrainerNode`'s
real loop into a shared function (`nodes/train/loop.py`) and built a
second trainer node, `BudgetedLoRATrainerNode`, that called the exact
same function with two different Port defaults (`resource_control`
required, `empty_cache_every_n_steps` defaulting to 50). Both are gone
now -- correctly identified as having missed the actual point: sharing
one loop meant the "new route" was the main route's loop with a
stricter gate in front of it, not an alternative memory design. The
reasoning used to justify *not* offloading model/optimizer
(`nothing calls ensure_loaded() on either mid-run, so marking them
offloadable would let before_step() offload one and never bring it
back`) was correct for that shared loop, but was never re-examined for
whether the new route actually had to accept the same constraint --
it doesn't, and doesn't now (see "The actual design" below).

### `TrainerResourcesUnpackNode`, also reverted

The first version also added a small node adapting `trainer` into the
main route's own `model`/`text_encoder` ports, specifically so nothing
about `TrainerNode`/`ModelParametersNode`/`CachingTextEncoderNode` would
need to change. That reasoning was sound for making the bundle wireable
into the *main route's* ports, but Phase 9's own trainer node no longer
has those ports to adapt into -- it takes `trainer` directly (see
below), so there's nothing left for that node to bridge. Its role for
the optimizer side is now `TrainerParametersNode`
(`nodes/model/trainer_parameters.py`) -- `ModelParametersNode`'s own
counterpart for this route, extracting a `ParameterList` straight from
`trainer.unet.trainable_parameters()`.

### The actual design: `nodes/train/managed.py`

A new, independent step loop (`ManagedStepPhase`/`ManagedTrainingStepPipeline`/
`ManagedLoRATrainerNode`) -- not importing from, or shared with,
`nodes/train/step_pipeline.py`/`nodes/train/supervised.py`. Written
using those files as a reference for the underlying math (diffusion
input prep, forward, loss, backward -- unrelated to what actually
differs here, so pointless to re-derive from nothing), not as a base to
extend or a shared implementation to call into.

**What's actually different.** The main route keeps every resident
loaded for a step's entire duration by default, offloading only
reactively, under measured pressure. This route's loop instead treats
residency as deterministic: each resident is loaded immediately before
the one phase that needs it and released immediately after, every step,
budget exceeded or not.

- `EncodeConditioningPhase`: loads `text_encoder`, encodes, releases it
  -- correct unconditionally, not just an optimization, because the
  text encoder is always frozen in this design (`TrainerParametersNode`
  never pulls trainable parameters from it), so nothing downstream
  needs it resident once `ctx_emb`/`y` are computed.
- `BackwardAndOptimizerStepPhase`: loads `optimizer`, runs backward
  (and `.step()` for a non-fused optimizer), releases it -- one phase,
  one residency window spanning both, not two. Checked directly against
  `ComposedFusedOptimizerHandle` (`nodes/optimizer/composed_fused.py`):
  a fused optimizer's real update happens inside a backward-pass hook
  (`_on_grad_ready()`, fired per-parameter as each gradient becomes
  ready *during* `backward()` itself), not in a separate call after --
  `FusedOptimizerHandle.step()` is a no-op. So optimizer state has to
  already be resident before `backward()` starts, for a fused optimizer
  -- not just before some later step()-shaped phase. A non-fused
  optimizer doesn't strictly need state resident during backward, only
  during its own `step()` call after, but loading it slightly earlier
  than strictly required is a small, deliberate over-inclusion, traded
  for one rule correct for both cases rather than a fused/non-fused
  branch in the residency logic itself.
- `model`: registered `offloadable=False`, same conclusion the main
  route reaches, reached independently this time -- the frozen base
  dominates the model's own footprint and is too expensive to move
  every step (a real multi-GB transfer, likely making per-step
  offload/reload slower than the run it's meant to protect).

Why this is a real, not token, difference for an SDXL LoRA run
specifically: the optimizer's tracked state is proportional only to the
trainable LoRA parameters, a small fraction of the base model's size --
cheap to move every step. SDXL's two text encoders (a full CLIP
ViT-L/14 plus OpenCLIP ViT-bigG/14) are large enough on their own that
keeping them off GPU outside their one phase is a real reduction for
the step's most memory-hungry stretch (forward+backward, activations
included) -- not just visibility into usage, an actual smaller peak.
Whether the transfer cost this adds is worth it against the main route
is exactly what this route exists to let someone measure, not something
decided here.

`ResourceControlHandle.release()` (`nodes/memory/control_handle.py`) is
new this session, specifically for this file -- the existing handle
only ever offloaded reactively, with no way for a caller who already
knows precisely when a resident's idle window starts to just say so.
Raises if called on a resident registered `offloadable=False`, same
"an explicit request to move something is not a hint" reasoning as the
rest of this handle's contract. Everything else this file uses from
`nodes/memory/` -- `register()`/`ensure_loaded()`, `ResourceCoordinator`,
`DeviceResident` -- is unchanged, already-established machinery, used
here as directly as the main route uses it. `before_step()` still runs
every step too, as a safety net for `model` alone exceeding the budget
-- this is exactly where `strict=True` (below) matters: without it,
that case would train on silently over budget forever, since nothing
in this design would ever offload `model` to bring it back under.

`resource_control` is required on `ManagedLoRATrainerNode`, not
optional -- this design has nothing meaningful left to do without it.
`empty_cache_every_n_steps` defaults to 1 (every step), not the main
route's 0: a `release()` call only moves tensors off GPU inside this
process's own caching allocator -- returning that freed reserved memory
to the driver, the actual point for a route built around staying clear
of a VRAM ceiling, needs an explicit `empty_cache()` on top.

### VRAM safety: `strict`, and an explicit `synchronize()`

Both in `nodes/memory/control_handle.py` (shared, unchanged from the
first version of this phase -- see that module's own docstring for the
full reasoning, unaffected by the revert above):

1. **`ResourceBudget.strict`** (`nodes/resource_budget.py`, default
   `False`). `_make_room()` previously offloaded what it could and
   returned, over budget or not, silently. `strict=True` raises
   `RuntimeError` instead, once nothing registered offloadable is left
   to move and measured usage is still over budget. Exposed as
   `VRAMBudgetControllerNode`'s `strict` input
   (`nodes/memory/vram_budget_controller.py`).
2. An explicit `DeviceContext.synchronize()` after every offload/reload/
   release transition, before trusting the next `memory_stats()` read.
   Not provably necessary from this codebase's own `offload()`/`reload()`
   implementations alone (every one already does a synchronous
   `.to("cpu")`/`.cpu()`, checked directly, not assumed) -- added
   anyway: this project's own legacy `core/trainer.py` (same B580/XPU
   hardware) already learned the hard way that even a nominally-
   synchronous transfer is worth an explicit `synchronize()` before
   trusting a memory snapshot right after it, at the exact kind of
   offload point `docs/known-issues/open.md`'s "device lost"/hang
   report names as a real trigger. A plausible root-cause *shape*, not
   a confirmed diagnosis -- see that file's own entry and
   `docs/known-issues/pending-testing.md`'s new one.

### Verified

A focused run, not the full 60-file suite -- markdown/plan edits don't
need a test run at all, and re-running every pre-existing test after
touching one new, independent module doesn't tell you anything the
tests for that module and its direct dependents don't already:
`smoke_test_managed_trainer.py` (new -- the actual thing under test:
that optimizer/text_encoder residency really is bracketed in the right
order around encode/backward/step for both a plain and a fused
optimizer, that model is never released, checked against real event
ordering, not assumed from reading the code), plus
`smoke_test_resource_control_strict.py` and
`smoke_test_text_encoder_cache.py` (both touched directly -- the former
gained a `release()`-specific check, the latter's own
`ResourceControlHandle` test fixture needed the new abstract method
implemented to stay instantiable). All three pass. `TrainerParametersNode`
(`nodes/model/trainer_parameters.py`) has no dedicated test -- it's a
three-line extraction with no branching, the same size and shape as
`ModelParametersNode`, which has never had one either.

### Not verified

No real XPU/CUDA hardware in this environment -- the residency
choreography's correctness (call order, which resident, which window)
is checked; its actual effect on peak VRAM, and on step time, is not
measured anywhere in this repository. Whether the `synchronize()`
addition helps with, or is even related to, the real "device lost"/hang
reports in `docs/known-issues/open.md` is unconfirmed and stays
unconfirmed here. `unet_weight_store="nf4"`'s path through
`LoRATrainingConfigNode` -- the complementary lever for the model's own
footprint this design deliberately doesn't try to shrink via offload --
isn't newly touched or newly tested by this phase either.

**Dependency:** Phase 6 (done).

## Addendum: from "always release" to "release only if measured usage actually needs it"

A real run reported ~0.36 steps/sec against this route vs. ~1.7 steps/sec
on the main route (same settings, AdamW) -- a ~4.7x slowdown -- with
peak reserved VRAM around 9.0GB against a stated 12500MB budget the
whole time. Investigated rather than guessed at; found several real,
compounding causes, in roughly descending order of likely impact:

1. **`SDXLTextEncoder.offload()` routed through `unload()`
   (`core/clip_encode.py`), which calls `gc.collect()` +
   `empty_cache()` internally, every single call.** Appropriate for
   `unload()`'s own original "done with this encoder for the rest of
   the run" use, real, avoidable, previously-invisible cost for a
   per-step offload cycle. Fixed: `offload()` now does a direct move,
   no `unload()`. Regression test:
   `nodes/smoke_tests/smoke_test_sdxl_text_encoder_offload.py`.
2. **The original design released text_encoder/optimizer
   unconditionally, every step, regardless of whether the stated
   budget ever needed it.** In the reported run it never did -- every
   release()/ensure_loaded() round trip was pure cost, bought nothing.
   Fixed with `AdaptiveResidencyController` (`nodes/train/managed.py`,
   full reasoning in its own docstring): `calibration_steps` steps run
   fully resident first, measuring real peak VRAM
   (`DeviceContext.reset_peak_stats()`/`memory_stats()`'s own
   `peak_reserved_mb`, `nodes/components/device.py`); if that already
   fits the budget, nothing is ever released for the rest of the run.
   If it doesn't, candidates are released smallest-`footprint_bytes()`-
   first, only as many as the estimated shortfall needs.
   `resource_control.usable_budget_mb()` is a new
   `ResourceControlHandle` method (alongside `release()`) so the
   controller doesn't need to know how a budget is represented
   internally. Tested directly (fake numbers, no hardware needed) in
   `nodes/smoke_tests/smoke_test_adaptive_residency_controller.py`;
   the CPU-only integration tests in `smoke_test_managed_trainer.py`
   can only exercise the "no usable ceiling / no memory-stats concept
   -> decide immediately, stay resident" fallback (`DeviceContext.
   for_device()` on a CPU tensor returns `_NullDeviceContext`, whose
   `memory_stats()` is always `None`) -- both are drilled separately
   in the same file for the "controller actually decided to release,
   do the phases honor it" side.
3. **No candidate is wrapped in `CachingTextEncoder` on this route**
   (`trainer.clip` is always a plain `SDXLTextEncoder`) -- for a
   dataset with repeated prompts (a single-image dataset, concretely,
   the reported case), the main route's own `CachingTextEncoderNode`
   wiring would make conditioning nearly free after the first step;
   this route pays full transfer + full recompute every step,
   unconditionally, with no possible cache hit. Real, likely
   significant for that specific case, **not fixed here** -- would need
   `trainer.clip` exposed as its own wireable output (the role
   `TrainerResourcesUnpackNode` used to play, before this file's own
   "First attempt, reverted" section, for a different reason). A real,
   scoped, separate follow-up, not attempted in this same change.
4. **No pinned (page-locked) host memory anywhere in this project's
   offload/reload paths** -- already known and disclosed, not new:
   `03-training-step-orchestration.md` section 2.3 already flags this
   exact gap ("a real platform-specific wrinkle... left for its own
   follow-up"). Still real, still unaddressed, now more likely to
   matter given how much more offload/reload traffic this route can
   generate.
5. **A rank-64 LoRA's own optimizer state is not negligible** --
   corrects an assumption in this doc's own first version ("proportional
   only to the trainable LoRA parameters -- a small fraction of the
   base model's size -- cheap to move every step"), true at low rank,
   not reliably true at rank 64. The offload/reload path itself has no
   hidden costs the way text_encoder's did (checked directly against
   `ComposedOptimizerHandle.offload_states_to_cpu()`/
   `reload_states_to_device()`, `nodes/optimizer/composed.py`) -- just
   real bytes moved, real PCIe time.

**Deliberately not attempted in this same change:** wiring gradient
checkpointing (`nodes/model/gradient_checkpointing.py`) or its own
budget-driven block-placement policy
(`GreedyRatioPlacement`/`BlockCost`, `nodes/model/checkpoint_placement.py`)
into this route as a genuine "slower but more memory-efficient"
alternative to offloading -- both already exist, real and working, and
that policy's own docs already flag it as unvalidated against a real
training run and deliberately not wired into any node's construction
path yet, for the same "measure, don't guess" reason
`AdaptiveResidencyController` exists. Extending that same caution to a
second, newer piece, built and tested on hardware neither has ever
actually run on, felt like the right call rather than rushing both in
under one patch. A natural next step, not started here.

## Second addendum: calibrate-once wasn't enough either

A second real report, from real use of the addendum above:
`AdaptiveResidencyController` calibrated fine (measured peak comfortably
under budget, decided to stay resident) and the run still OOM'd partway
through, on a variable-resolution ("non-square") dataset. Root cause:
each step's own activation memory depends on that step's own image
size, and `calibration_steps` (default 3) happened to sample smaller
images -- the true worst case in the dataset was never measured before
the "stay resident" decision was locked in for the rest of the run.

Two real, complementary fixes, not a bigger `calibration_steps` default
(which only ever narrows the odds, never closes the gap -- the largest
image in a dataset can be anywhere):

1. **Ongoing escalation, not calibrate-once.** `ManagedLoRATrainerNode.
   build()` now calls `DeviceContext.reset_peak_stats()` every step, not
   just once before calibration, so every `memory_stats()` reading
   reflects *that step's own* peak, not a cumulative one -- which is
   what makes "keep checking after the initial decision, and escalate
   (release one more candidate) if a later step's own peak exceeds
   budget" possible at all. No de-escalation once something's been
   added, to avoid thrashing. Honest limit, stated plainly in
   `AdaptiveResidencyController`'s own docstring: this still can't react
   *within* the step that actually OOMs -- a within-step activation
   spike isn't something any between-step check can catch in time, the
   same limit `before_step()`'s own reactive check already had. What it
   does buy: the *next* image of that size survives, once one instance
   has been seen and escalated for once.
2. **`residency_safety_margin`** (new `ManagedLoRATrainerNode` input,
   default 0.1): shaves that fraction off the usable ceiling before any
   comparison runs, so a somewhat-larger-than-calibrated step has a
   chance of fitting without needing to escalate at all.

Neither one is a guarantee for a dataset with truly wide resolution
variance and a small `calibration_steps` -- both are real, disclosed
insurance, not a solved problem. The right fix for large numbers of
same-answer variance would look at the dataset itself (bucket by
resolution and calibrate per bucket, or make calibration explicitly
seek out the largest sample) -- not attempted here.

## Third addendum: activation memory is the ceiling residency management and weight precision can't touch

The report that surfaced the above also included the actual numbers
behind an OOM: 10218MB reserved, of which the three residents this
route tracks accounted for only ~4806MB combined (model 3061MB +
optimizer 184MB + text_encoder 1561MB) -- more than half the real usage
was activation memory (forward/backward intermediate tensors), which
neither `AdaptiveResidencyController` nor NF4/Int8 weight quantization
touches at all. Releasing every candidate this controller manages
(1745MB combined, in that report) was never going to be enough headroom
against a multi-GB activation spike on its own.

The same report asked why switching from NF4 (frozen base) + Int8
(optimizer state) to bf16 + fp32 only changed measured usage by ~1GB,
much less than the ~4x difference their own storage footprints would
suggest. Real, and not a bug: both `NF4WeightStore` (`nodes/model/
nf4_weight_store.py`) and `Int8BlockStateStore` (`nodes/optimizer/
state_store.py`) dequantize to a real, transient full-precision buffer
on every use -- `footprint_bytes()` deliberately doesn't count that
buffer (both classes' own docstrings: it's the caller's, freed right
after, not storage either class holds) -- so the *resting* footprint is
genuinely ~4x smaller, but the *peak-during-compute* footprint (what
actually determines whether a step OOMs) is much closer to the
unquantized case, because the transient buffer still has to exist for
that one use. This is inherent to this whole class of technique
(bitsandbytes/QLoRA has the same characteristic), not specific to this
project's own implementation. (Also, for the record: `unet_weight_store`
only has two real choices, `"bf16"` and `"nf4"` --
`nodes/model/lora_training_config.py`'s own `_UNET_WEIGHT_STORE_CHOICES`
-- there is no `"nvfp4"` option in this codebase; a value outside those
two would have raised immediately via `Port.choices` validation, so
whatever was actually selected and produced these numbers was `"nf4"`.)

Together, these two findings point at the same conclusion the first
addendum already named as deliberately not attempted: gradient
checkpointing (`nodes/model/gradient_checkpointing.py`, real, working,
not wired into this route) is the lever that actually addresses
activation memory, the dominant, currently-unmanaged cost in both
reports. Two real cases now, not a hypothetical -- still not started
here; see the first addendum's own reasoning for why rushing it into
the same session as everything else felt like the wrong tradeoff, which
still holds, but the case for it being the actual next priority (over
further residency-management or precision tuning) is now real, not
speculative.

**Update, follow-up session:** checked directly, not assumed --
`use_checkpoint` (`ComfyUNetLoRANode`'s own Port) already defaults
`True` all the way through `build_lora_injected_unet()`, and nothing in
this route's own construction path (`LoRATrainingConfigNode.build()` ->
`SDXL_LoraTrainer.from_resources()`) overrides it, so gradient
checkpointing was, in that narrow sense, already active for both
reports above. It just wasn't doing much: `docs/known-issues/
pending-testing.md`'s `[2026-09]` entry has the full story --
`BasicTransformerBlock.forward()` never actually called `checkpoint()`
in ComfyUI's own implementation, so `use_checkpoint=True` only ever
checkpointed `ResBlock`, a minority of SDXL's UNet next to its
attention-heavy `BasicTransformerBlock` stacks. That's almost certainly
why activation memory stayed ~half of reserved VRAM in both reports
above despite checkpointing nominally being on. `nodes/model/
attention_checkpointing.py` closes that gap; not yet confirmed this
actually pulls either of the two real reports above under budget --
needs a real before/after run to know by how much.
