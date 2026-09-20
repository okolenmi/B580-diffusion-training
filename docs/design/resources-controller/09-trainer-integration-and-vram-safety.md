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
