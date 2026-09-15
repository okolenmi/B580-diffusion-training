# Progress

Fast-read summary of what's actually implemented in `nodes/` -- the
design-doc-driven rewrite of the training pipeline. For the rationale
behind any of this, see [`docs/design/`](../design/README.md) (that
doc was split into one file per top-level section during this docs
restructuring; the section numbers cited below, e.g. "(1.1)",
"(3.1)", still refer to the same section numbers, just look at
`docs/design/README.md`'s table to find which file a given number now
lives in). For known bugs, see
[`docs/known-issues/`](../known-issues/README.md) -- an informal
collection, not authoritative, mostly about the legacy `core/`
pipeline unless a `nodes/` path is named.

`core/` and `manager/` themselves are not modified by this rewrite --
they're the current production path, and stay that way. `nodes/` no
longer treats wrapping them as a permanent rule, though: where `nodes/`
has since built its own independent, verified-equivalent replacement
(the `optimizer/` domain, `components/`), that replacement is canonical
and the old wrapper gets retired -- see
[`docs/CLEANUP_TODO.md`](../CLEANUP_TODO.md) for what's been unified,
what's still mid-migration, and what hasn't started. Most of `nodes/`
still wraps `core/`/`manager/` directly (LoRA/UNet injection, text
encoding, dataset ingestion) simply because nobody's built an
independent version yet -- that's where new work lands.

## Implemented

**Foundational**
- `Node`/`Port` construction-time contract -- `nodes/core.py` (1.1).
  The design doc's own illustrative name for this concept was
  `Builder`; the real, pre-existing class is `Node` -- see
  `docs/design/08-validation-and-implementation-status.md`'s table,
  which gets this right.
- `DeviceResident` ABC (offload/reload/footprint), conformed to by
  `OptimizerHandle`, `TrainableModel`, `TextEncoder` --
  `nodes/memory/handle.py` (1.2)
- Pooled device buffers -- `nodes/memory/manager.py`'s `MemoryManager`
  (1.3)
- `NoiseSchedule`/`Parameterization`/`DiffusionProcess`, `DeviceContext`
  -- `nodes/components/diffusion.py`, `nodes/components/device.py`
  (1.4, 1.5)
- `ProjectLayout`, bridging `paths.py` -- `nodes/components/layout.py`
  (1.6)

**Resource policy**
- `ResourceBudget` -- `nodes/resource_budget.py` (2.2), used by
  `checkpoint_placement.py` and `nodes/memory/`.
  `ResourcePolicy`/`ManualResourcePolicy` (same section, formerly the
  same file) were removed: no `Node` ever produced a `ResourcePolicy`,
  so `ComfyUNetLoRANode`'s `resource_policy` port was unreachable from
  the graph editor -- only ever constructible by hand, in Python. The
  concerns it bundled (checkpointing strategy, LoRA scaling policy) are
  each still independently settable on `ComfyUNetLoRANode` directly.
  `Composed*OptimizerNode`'s `group_policy` port is a separate
  mechanism (`ParameterGroupPolicy`, `nodes/optimizer/composed.py`) and
  was never actually routed through `ResourcePolicy` despite this
  section once implying otherwise.
- `ActivationCheckpointingStrategy` --
  `nodes/model/gradient_checkpointing.py` (2.3)
- `BlockCost`/`CheckpointPlacementPolicy`/`EveryBlockPlacement`/
  `GreedyRatioPlacement` -- `nodes/model/checkpoint_placement.py` (2.3).
  `GreedyRatioPlacement` unvalidated against a real run.
- `BlockProfileCollector`/`ProfilingCheckpointing` --
  `nodes/model/block_profiler.py` (2.3). The instrumentation that was
  actually blocking the above. Real finding: only `ResBlock` instances
  route through this in the current ComfyUI version -- attention blocks
  never call `checkpoint()` at all. **Not wired** into
  `ComfyUNetLoRANode`'s ports yet.
- Text encoder cache as a `DeviceResident` -- `nodes/model/text_encoder.py`,
  `text_encoder_cache.py` (2.4)
- `PrefetchingBatchSource` -- `nodes/dataset/prefetch.py` (2.5)

**Model / LoRA**
- `AdapterStrategy`/`PlainLoRAAdapter`/`DoRAAdapter`, `LoRAScalingPolicy` --
  `nodes/model/adapter_strategy.py`, `nodes/model/dora_layer.py`,
  `nodes/model/lora_scaling.py` (3.1, 3.2). **Live-wired** into
  `ComfyUNetLoRANode`'s real construction path via
  `nodes/model/adapter_injection.py`'s `adapter_strategy_scope` (new
  `adapter_strategy` port, default `None` -> `PlainLoRAAdapter()`, no
  behavior change). `DoRAAdapter` grounded directly in HuggingFace
  PEFT's real source (fetched and read, not recalled). Fixed a real
  recursion hazard along the way (see `lora_class_cache.py`'s
  `_real_lora_classes()`). Checkpoint save/load (direction +
  `.dora_scale` magnitude + alpha, key name matching ComfyUI's own
  `comfy/lora.py` convention) now real for an unsplit DoRA layer, via
  `nodes/model/lora_phases.py`'s `extract_combined_weights`/
  `extract_own_generation_weights` and `LoRACheckpointLoaderNode`'s new
  `_load_dora_layers()`. That loader fix closed a deeper pre-existing
  bug found in the process: `core.lora.load_lora_into_model`'s
  isinstance gate silently skipped every DoRA layer entirely (not just
  its magnitude -- direction and alpha too), since `DoRALinear`/
  `DoRAConv2d` are built via composition, not inheritance, over
  `core.lora.LoRALinear`/`LoRAConv2d`. Also found and fixed:
  `split_into_new_generation` (and `LinearLoRAGeneration`/
  `Conv2dLoRAGeneration._build_params()`) assumed that same attribute
  layout when freezing a generation, so phase-splitting a DoRA layer had
  never actually worked at all -- raised `AttributeError` -- independent
  of the checkpoint question; fixed by freezing through
  `get_lora_weights()` instead, plus explicitly freezing `magnitude`
  (not reachable through that call), verified with a real
  gradient-isolation check. **A second pass, deliberately looking for
  more instances of the same "composition, not inheritance" pattern,
  found the most severe one yet: a real DoRA training run trained
  nothing at all.** `core.unet_wrapper.ComfyUNetWrapper._init_lora()`
  freezes every parameter then re-enables `requires_grad` only where
  `hasattr(layer, "lora_A")` -- False for a bare DoRA layer, so
  `lora_A`/`lora_B`/`magnitude` all stayed frozen silently. Fixed by
  `adapter_injection.py`'s new `reenable_dora_requires_grad()`. Not
  sufficient alone: `ComfyUNetWrapper.lora_parameters()` (what
  `ComfyUNetTrainableModel.trainable_parameters()` hands to the
  optimizer) has the identical gate, so the optimizer would never have
  received a single DoRA parameter regardless of `requires_grad`. Fixed
  by the new `dora_trainable_parameters()`, combined into
  `trainable_parameters()`/`footprint_bytes()` (the latter was also
  silently miscounting DoRA's own tensors as part of the frozen base).
  Verified with a real two-step optimizer loop that provably moves
  every DoRA parameter, not just `requires_grad` reading `True`. One
  real gap left, narrower and honestly open: a phase-split DoRA layer's
  magnitude still can't be folded into a *combined* checkpoint --
  `extract_combined_weights` raises a clear error for that case rather
  than silently dropping the trained magnitude. See
  `docs/design/04-lora-adapter-mechanics-and-loss-weighting.md`
  (section 3.1) and
  `docs/design/08-validation-and-implementation-status.md` (section 9.2).
- `FrozenWeightStore`/`BF16WeightStore`/`NF4WeightStore` --
  `nodes/model/frozen_weight_store.py`, `nodes/model/nf4_weight_store.py`,
  `nodes/model/nf4_lora_layer.py` (3.3). NF4 quantization grounded
  directly in bitsandbytes' real source (3.875x compression vs bf16 at
  realistic scale). **Wired** into a real forward path --
  `NF4LoRALinear`/`NF4LoRAConv2d`, a `frozen_weight_store` port on
  `ComfyUNetLoRANode`. Still needs a real-run quality check.
- `ParameterGroupPolicy`, `LoRAPlusGroups` --
  `nodes/optimizer/composed.py` (3.4). Selectable from the graph now,
  **not yet validated** with a real tuned run.

**Training loop**
- `TrainingStepPipeline`/`StepPhase` -- `nodes/train/step_pipeline.py`
  (2.1)
- Min-SNR v-prediction branch + `P2LossWeighting` --
  `nodes/train/loss.py` (section 4)
- LoRA timestep gate (`gate_enabled`/`gate_train_low`/`gate_train_high`/
  `gate_width`) wired into `PrepareDiffusionInputsPhase` -- candidate fix
  for a real deformation report, **not yet run** on real data (see
  `docs/known-issues/pending-testing.md`)

**Memory / offload**
- `ResourceCoordinator`/`OffloadOrchestrator` --
  `nodes/memory/coordinator.py` (5.1, 5.2). Doesn't by itself fix the
  open VRAM-hang report against `core/trainer.py`. Still exactly as
  un-wired as ever -- event-driven for three specific, rare moments
  (cache rebuild, preview generation, checkpoint save), nothing in the
  real training loop publishes those events yet.
- `ResourceProfile` -- `nodes/memory/profile.py` (5.5). Per-`DeviceResident`
  VRAM breakdown, wired into `profile=True`'s existing report
  (`resident_<name>_mb` alongside `tracked_footprint_mb`). Real gap
  found while landing this, not yet fixed: no shared `MemoryManager`
  reachable from the trainer node, so `memory_manager_stats` is always
  `None` in a real run today -- see the module docstring.
- **`ResourceControlHandle`/`BudgetedResourceControlHandle` -- a live,
  per-step VRAM budget enforcer, new since the list above and
  genuinely different from `OffloadOrchestrator`, not a rename of it.**
  `nodes/memory/control_handle.py`. Not "react to a named, rare
  event" (that's what `OffloadOrchestrator` already does, unwired) but
  "check real measured usage before every step, offload whatever's
  marked safe to if over budget, reload it right before whatever needs
  it next actually needs it." A handle one node constructs
  (`VRAMBudgetControllerNode`) and another (the trainer) calls into
  during its own `build()` -- same shape `MonitorHandle`/
  `LiveMonitorHandle` already established, not a second graph node
  running "alongside" the trainer (`server/graph_executor.py` runs
  nodes one at a time; there's no mechanism for two to exchange live
  signals mid-execution). Wired into `SupervisedLoRATrainerNode` via a
  new `resource_control` input. Honestly incomplete in one specific
  way: `model`/`optimizer` are never marked offloadable, since neither
  has a genuine idle window in this pipeline's always-synchronous
  design, and nothing yet calls `ensure_loaded("model")`/
  `ensure_loaded("optimizer")` at the right point to make offloading
  either safe. What *is* wired end to end: `CachingTextEncoder`
  (`nodes/model/text_encoder_cache.py`) takes an optional
  `resource_control` -- on a cache miss it calls `ensure_loaded()`
  before falling through to the inner encoder, so it's always safe for
  the inner encoder to have been offloaded between hits.
  `SupervisedLoRATrainerNode` marks `text_encoder` offloadable exactly
  when it's actually a `CachingTextEncoder` (checked via `isinstance`).
  `ensure_loaded()` itself shares a `_make_room()` helper with
  `before_step()`, so reloading something that would push usage over
  budget offloads other offloadable residents first -- direct handling
  of "model/optimizer already near budget when a cache miss needs the
  text encoder back."
- **8-bit optimizer-state quantization, `state_precision`** (design
  section 11.3, this item was originally scoped as a plain bf16 cast --
  shipped instead as something better-validated). `OptimizerStateStore`/
  `Int8BlockStateStore` (`nodes/optimizer/state_store.py`) block-wise
  quantizes `m`/`v` to 8 bits between steps (dequantize to real fp32 ->
  `Algorithm.compute_update()` runs completely unchanged, unaware this
  exists -> requantize the result) -- ~4x smaller than a bf16 cast's 2x,
  and actually verified end to end (a real 20-step AdamW comparison,
  `Float32StateStore` vs `Int8BlockStateStore`, same seed/gradients,
  converges within 0.0025 max per-parameter difference), where the
  original bf16-cast idea was flagged as an unvalidated numerical risk
  and never shipped. One shared implementation: `state_precision`'s
  choices/doc/resolver live once in `state_store.py`, reusing
  `strategy_registry.py`'s `STRATEGIES`/`resolve_strategy()` shape for
  a different Port on the same three optimizer nodes. Lives on the
  `Composed*` optimizer nodes themselves, same place `strategy`/`device`
  already did -- the Resources Controller redesign (below) considered
  and explicitly decided against absorbing this into its own precision
  handling (see `docs/design/resources-controller/08-consolidation.md`).

**Resources Controller / precision redesign** -- the most recently
active work in the repo; full detail and current status in
[`docs/design/resources-controller/`](../design/resources-controller/README.md),
condensed here. Phases 1 and 2 (lazy `ModelWeights`/
`SafetensorsCheckpointNode` header-only inspection; a server query
endpoint for checkpoint dtype) and Phase 3 (`Node.NODE_KIND`/
`NodePreset`/`list_presets()` -- generic infrastructure any node can
use to be found by the editor's suggestion-menu search) are done.
Phase 4's `ResourcePreset` construction mechanics are done:
`SDXLArchitecture`+`LoRATrainingSkeleton` compose (multiple
inheritance, concrete-mixin-first) into `SDXL_LoraTrainer`, a real
`DeviceResident` built on the existing `ResourceCoordinator`. Phase 5's
`ResourcesControllerNode` is done, scope-corrected along the way (an
earlier version did LoRA injection itself; corrected to produce only a
verified, NOT-yet-injected `LoRATrainingResources` pack -- injection is
a property of a training config, not a verified resource). Phase 5 also
landed generic editor mechanics any node can use, not just this one:
`Port.choices` (closed-choice dropdown, e.g. `strategy`/`device`
fields that used to be free-text strings), `Port.visible_when`
(conditional port visibility), `Port.widget_only` (a checkbox that
doesn't need its own wire socket), and a live `Node.diagnostics()`
endpoint. Phase 6's `LoRATrainingConfigNode` is done: takes Phase 5's
resource pack and actually injects LoRA (rank/alpha/frozen-weight-
storage), including locking rank when continuing training from an
existing LoRA file. `SDXL_LoraTrainer` (Phase 4) also supports two
distinct LoRA-file inputs, not just one: `frozen_lora_sd` merges a saved
LoRA directly into the base weights at load time before injection
(`nodes/model/lora_merge.py`'s `merge_lora_into_state_dict`, no separate
object -- it has no identity afterward, just changed base weights), while
`continue_lora_sd` loads a saved LoRA into the new trainable adapter
itself, to actually resume training it (reuses
`load_lora_into_registry()`, extracted from `LoRACheckpointLoaderNode`
for this). **`TrainerNode` integration is the one piece still
open** -- nothing under `nodes/train/` references
`LoRATrainingConfigNode`/`LoRATrainingResources` yet (checked
directly), so the config node's output has nowhere to actually plug in
today.

**Server / graph**
- `server/graph_executor.py` -- topological execution, port-compatibility
  checking, `ExecutionContext` threading; pre-existing, matches the
  design's construction-time model already.
- `server/nodegraph_registry.py` -- palette list; currently matches every
  concrete `Node` subclass in `nodes/`.

**Testing**
- 56 smoke tests under `nodes/smoke_tests/` (runnable via
  `nodes/smoke_tests/run_all.py`), plus 5 more under `server/` and 1
  under `manager/` -- all CPU-only, no ComfyUI/XPU needed.

## Still open, in priority order

See `docs/design/09-prioritized-backlog.md` (section 10) for the full
reasoning behind this order. Two items below are newer than that
backlog and not yet folded into its own ordering:

1. Validation only, code already exists: `RescaledZeroTerminalSNRSchedule`
   end-to-end training run (1.4); `LoRAPlusGroups` actually tuned against
   a `UniformGroups` baseline (3.4); `GreedyRatioPlacement` wired into
   `ComfyUNetLoRANode` and run against a real profiled `BlockCost` set
   (2.3); `DoRAAdapter` run on real data to confirm the quality
   improvement shows up here too (3.1); `NF4WeightStore`'s
   diffusion-specific quality check against this project's real UNet
   (3.3)

**Newest, not yet prioritized against the list above:**
`LoRATrainingConfigNode`/`LoRATrainingResources` (Resources Controller
Phase 6) have nothing under `nodes/train/` to plug into yet -- wiring
them into `TrainerNode` is real, scoped work, not started. Also:
`ResourceControlHandle` only ever marks `text_encoder` offloadable
today -- extending that to `model`/`optimizer` needs somewhere in the
step pipeline to call `ensure_loaded("model")`/`ensure_loaded("optimizer")`
at the right point first, which doesn't exist yet either.

**Not yet its own item, nothing above needs it yet:** thread a shared
`MemoryManager` through optimizer construction so `ResourceProfile`'s
`memory_manager_stats` is ever populated in a real run -- see
`nodes/memory/profile.py`'s module docstring. Also: a phase-split DoRA
layer's magnitude can't be folded into a *combined* checkpoint --
`extract_combined_weights` raises a clear error rather than silently
dropping it (see `docs/design/08-validation-and-implementation-status.md`
section 9.2) -- real, but nothing currently in this project's own
recommended workflows actually needs a phase-split DoRA layer combined
this way.

**Not recommended near-term:** `ComponentRegistry`/`TrainingRecipe`/
`PipelineFactory` (5.3, 5.4) -- nothing built through
`nodes/components/` so far is graph-editor-selectable, so the problem
these solve hasn't materialized.

**Deferred or rejected**, reasoning in full in
`docs/design/07-deferred-or-rejected.md` (section 7):
`AutoResourcePolicy`, automatic eviction inside `MemoryManager`,
layer-wise base offload, flow matching, GaLore, 8-bit optimizer moments.

---
Last synced against `docs/design/` (formerly the single file
`docs/training_pipeline_design.md`) at commit `2991618` (2026-09-10,
"optimizer: state_precision -- block-wise 8-bit quantized optimizer
state") -- the last substantive feature commit before the docs
restructuring. Resynced from the previous sync point, which claimed
commit `2c1f0ff` (2026-08-25) but no longer resolves to a real object
in this repository's history as of this resync -- likely a rewritten
commit from before this clone's history; not investigated further
since the content gap it left (everything from the Resources
Controller redesign's Phase 1, 2026-08-26, onward) was fully
recoverable by date instead.
