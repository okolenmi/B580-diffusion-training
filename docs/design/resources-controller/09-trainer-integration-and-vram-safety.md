*[← Resources Controller index](README.md) · [docs/design index](../README.md)*

## Phase 9 -- `TrainerNode` integration, and VRAM safety as a second route's actual point

**Goal:** unblock the one thing Phase 6 left open (`LoRATrainingConfigNode`'s
`trainer` output had nowhere to plug in), and build the Resources
Controller route as a genuine second, memory-optimized alternative to
the existing main route -- not a replacement for it, a real alternative
to be run side by side and judged on whether it's actually better,
concretely including a way to cap VRAM usage that raises rather than
silently running past it.

**Status: done, verified against the real smoke-test suite (60/60
passing -- 56 pre-existing plus 4 new, run for real in this session,
not reasoned about untested), not yet run on real hardware. See "Not
verified" below for exactly what that leaves open.**

### `TrainerResourcesUnpackNode` -- the piece that actually unblocks the route

`nodes/model/trainer_unpack.py`. Takes `LoRATrainingConfigNode`'s
`trainer: LoRATrainingSkeleton` output, exposes `.unet`/`.clip` as
plain `model: TrainableModel`/`text_encoder: TextEncoder` outputs.
Deliberately not what `06-phase-6-lora-training-config.md` originally
sketched (replacing `TrainerNode`'s own ports with one bundled
`trainer` port) -- see that file's own updated status for exactly why
that would have broken the existing main route. This way, nothing
about `TrainerNode`, `ModelParametersNode`, or `CachingTextEncoderNode`
needed to change at all: wire `LoRATrainingConfigNode.trainer` into
this node, then wire its two outputs into exactly the same ports the
main route already fills from `ComfyUNetLoRANode`/`SDXLTextEncoderNode`.
`vae_sd`/`lora` aren't exposed -- nothing under `nodes/train/` needs
either today (checked directly: no VAE-decode/preview node exists yet
under `nodes/`).

### `BudgetedLoRATrainerNode` -- the route's actual memory-optimization story

Direct ask this session started from: "a way to set a limit of VRAM
usage, so I don't hit a point where my GPU may randomly fail." That
mechanism (`ResourceControlHandle`/`VRAMBudgetControllerNode`,
`nodes/memory/control_handle.py`) already existed from earlier work --
the real gap was that it was fully optional and easy to leave unwired,
and that model/optimizer aren't offloadable (needed every step,
nothing calls `ensure_loaded()` on either mid-run yet -- extending
that needs step-pipeline integration points that don't exist, tracked
in `docs/status/progress.md`'s own "still open" list, genuinely out of
scope here, not attempted).

`nodes/train/budgeted.py`: a second concrete `TrainerNode`, alongside
`SupervisedLoRATrainerNode`, not instead of it. Two Port-level
differences: `resource_control` required instead of optional, and
`empty_cache_every_n_steps` defaults to 50 instead of 0 (periodic
`gc.collect()`/`empty_cache()`, real small step-time cost, traded for
keeping this run's own peak reserved footprint further from whatever
ceiling `resource_control` enforces). Everything else -- gating,
profiling, `diffusion_process`, `loss_weighting`, `monitor`, `on_step`
-- is identical, and *actually* identical, not "kept in sync by hand":
both nodes now resolve their own Port defaults and call the same
`run_supervised_lora_training_loop()` (`nodes/train/loop.py`, extracted
from `SupervisedLoRATrainerNode.build()`'s previous inline body this
session, same shape as `nodes/model/lora_injector.py`'s
`build_lora_injected_unet()` extraction and the same reasoning
`08-consolidation.md`'s "Real redundancy risk found" section already
gives for that one). `BudgetedLoRATrainerNode` subclasses `TrainerNode`
directly, not `SupervisedLoRATrainerNode` -- every concrete `Node` in
this codebase subclasses its domain ABC directly and shares real logic
through a plain object instead (`ComposedFusedAdamWOptimizerNode`
next to `ComposedAdamWOptimizerNode`, sharing `AdamWAlgorithm`/
`ComposedOptimizerHandle` rather than one inheriting the other) --
`loop.py`'s extraction follows that same pattern rather than being the
one place a concrete `Node` inherits from another.

Route-agnostic on purpose: `model`/`text_encoder` are the same plain
ports `SupervisedLoRATrainerNode` already has, so `BudgetedLoRATrainerNode`
works from either route (wire directly from `ComfyUNetLoRANode`/
`SDXLTextEncoderNode`, or from `TrainerResourcesUnpackNode` on the new
route) -- it's "the trainer that mandates a VRAM budget," not "the new
route's trainer" specifically, and a person can use it on today's main
route too if they want the same guarantee there.

### VRAM safety: `strict`, and an explicit `synchronize()`

Both in `nodes/memory/control_handle.py` (shared by both routes, not
duplicated for the new one):

1. **`ResourceBudget.strict`** (`nodes/resource_budget.py`, default
   `False`, current best-effort behavior unchanged unless opted in).
   `BudgetedResourceControlHandle._make_room()` previously offloaded
   what it could and returned, over budget or not, silently. `strict=True`
   raises `RuntimeError` instead, once nothing registered offloadable is
   left to move and measured usage is still over the budget -- a hard
   stop instead of training on past the ceiling that was asked for,
   which is exactly the condition a budget exists to prevent. Exposed
   as `VRAMBudgetControllerNode`'s new `strict` input
   (`nodes/memory/vram_budget_controller.py`).
2. An explicit `DeviceContext.synchronize()` right after every
   offload/reload transition `_make_room()`/`ensure_loaded()` drive,
   before trusting the next `memory_stats()` read. Not provably
   necessary from this codebase's own `offload()`/`reload()`
   implementations alone -- every one of them already does a
   synchronous (`non_blocking=False`, torch's default) `.to("cpu")`/
   `.cpu()`, checked directly across `nodes/model/lora_injector.py`,
   `nodes/model/text_encoder.py`, `nodes/optimizer/composed.py`, not
   assumed. Added anyway: this project's own legacy `core/trainer.py`
   (same B580/XPU hardware) already learned the hard way that even a
   nominally-synchronous transfer is worth an explicit `synchronize()`
   before trusting a memory snapshot right after it -- that file's own
   comment, at the exact preview-generation offload point
   `docs/known-issues/open.md`'s "device lost"/hang report names as a
   real trigger. That same known-issues entry cites a matching report
   from different training code on this hardware
   (`kohya-ss/musubi-tuner`), tracing a comparable hang to a missing
   synchronize on an XPU offload path -- a plausible root-cause *shape*,
   not a confirmed diagnosis, and that entry explicitly scoped itself to
   `core/trainer.py`, not `nodes/`. This is cheap insurance in code that
   report hadn't reached yet, not a claim of having root-caused or fixed
   it -- see `docs/known-issues/pending-testing.md`'s new entry for the
   honest version of this.

`BudgetedResourceControlHandle`'s constructor also gained an optional
`device_ctx` parameter (default `None` -> today's real
`DeviceContext.for_device(device)`, unchanged) specifically so a test
could inject a fake one and script `memory_stats()` -- there was no way
to exercise `_make_room()`'s real threshold/offload-order logic at all
otherwise (see "Real gaps found" below).

### Real gaps found while doing this, fixed alongside it

Two smoke-test gaps, found by checking directly (grepped
`nodes/smoke_tests/` and `git log --all`, not assumed):

- **No smoke test ever existed for `LoRATrainingConfigNode`** --
  `06-phase-6-lora-training-config.md`'s own "Verified manually" section
  describes real checks that were never captured as a repeatable test,
  unlike every other phase. Fixed:
  `nodes/smoke_tests/smoke_test_lora_training_config.py`, covering
  dispatch, an unregistered-resources-type error, and the rank-locking
  behavior that section calls out as this node's one real, concrete job
  (a continuing LoRA's own detected rank overriding a different
  explicit `rank` input, and a malformed multi-rank `continue_lora_sd`
  being rejected before injection is attempted).
- **No smoke test ever exercised `BudgetedResourceControlHandle`'s real
  `_make_room()` logic** -- `smoke_test_text_encoder_cache.py`'s own
  `_RecordingResourceControl` only ever stood in for the
  `ResourceControlHandle` ABC. Fixed:
  `nodes/smoke_tests/smoke_test_resource_control_strict.py`, using the
  new `device_ctx` injection point above to script real threshold
  crossings, offload-order, `strict` raising vs. not, and
  `synchronize()` call counts.

Two more new nodes, `smoke_test_budgeted_trainer.py` and
`smoke_test_trainer_unpack.py`, check the two new node classes
specifically (required-port enforcement, default differences, real
behavioral parity with `SupervisedLoRATrainerNode`; real identity
forwarding against an actual constructed `SDXL_LoraTrainer`).

### Verified

Ran the full suite for real this session (`nodes/smoke_tests/run_all.py`,
60/60 passing), not reasoned about statically -- including every
pre-existing test, confirming the `nodes/train/supervised.py` extraction
changed nothing observable about `SupervisedLoRATrainerNode`'s own
behavior. Specifically real, not mocked: `smoke_test_resource_control_strict.py`'s
threshold/offload-order/strict/synchronize checks run
`BudgetedResourceControlHandle`'s actual code against a scripted (not
real-hardware) `DeviceContext`; `smoke_test_lora_training_config.py`'s
rank-locking check runs `LoRATrainingConfigNode.build()`'s actual
dispatch and rank logic against a really-constructed
`SDXL_LoRATrainingResources`, LoRA injection itself mocked only where
real ComfyUI internals would be needed (same boundary every
UNet-touching smoke test in this project already has).

### Not verified

No real XPU/CUDA hardware in this environment -- `DeviceContext.memory_stats()`
returns `None` on CPU, so `strict`/`synchronize()` are verified against
scripted `memory_stats()` values, not a real VRAM-pressure event. The
actual question this work was motivated by -- whether the
`synchronize()` addition helps with, or is even related to, the real
"device lost"/hang reports in `docs/known-issues/open.md` -- is
unconfirmed and stays unconfirmed here; see that file's own careful
hedging (root-cause *shape*, not diagnosis) and the new
`docs/known-issues/pending-testing.md` entry. `unet_weight_store="nf4"`'s
path through `LoRATrainingConfigNode` isn't covered by the new smoke
test (needs bitsandbytes-level mocking `06-phase-6-lora-training-config.md`'s
own manual verification didn't build either) -- untouched by this
session, not newly broken, just still not under test.

**Dependency:** Phase 6 (done).
