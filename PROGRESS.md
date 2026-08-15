# Progress

Fast-read summary of what's actually implemented in `nodes/` -- the
design-doc-driven rewrite of the training pipeline. For the rationale
behind any of this, see `docs/training_pipeline_design.md` (section
numbers below point there). For known bugs, see
`docs/suspicious_findings.md` -- an informal list, not authoritative,
mostly about the legacy `core/` pipeline unless a `nodes/` path is named.

`core/` and `manager/` are the current production path and are
deliberately untouched by this rewrite (wrap-don't-copy, per the design
doc's own rule) -- `nodes/` is where new work lands.

## Implemented

**Foundational**
- `Builder`/`Port` construction-time contract -- `nodes/core.py` (1.1)
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
- `ResourceBudget`/`ResourcePolicy`/`ManualResourcePolicy` --
  `nodes/resource_policy.py` (2.2). Wired into `ComfyUNetLoRANode`
  (`resource_policy` port) and all three `Composed*OptimizerNode`
  classes (`group_policy` port).
- `ActivationCheckpointingStrategy` --
  `nodes/model/gradient_checkpointing.py` (2.3)
- Text encoder cache as a `DeviceResident` -- `nodes/model/text_encoder.py`,
  `text_encoder_cache.py` (2.4)
- `PrefetchingBatchSource` -- `nodes/dataset/prefetch.py` (2.5)

**Model / LoRA**
- `AdapterStrategy`/`PlainLoRAAdapter`, `LoRAScalingPolicy` --
  `nodes/model/adapter_strategy.py`, `lora_injector.py` (3.1, 3.2). Seam
  built and equivalence-tested, **not live-wired** into
  `ComfyUNetLoRANode`'s real construction path yet (still
  `core.lora._inject_lora`).
- `FrozenWeightStore`/`BF16WeightStore` --
  `nodes/model/frozen_weight_store.py` (3.3)
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
  `docs/suspicious_findings.md`, "Pending user testing")

**Memory / offload**
- `ResourceCoordinator`/`OffloadOrchestrator` --
  `nodes/memory/coordinator.py` (5.1, 5.2). Doesn't by itself fix the
  open VRAM-hang report against `core/trainer.py`.
- `ResourceProfile` -- `nodes/memory/profile.py` (5.5). Per-`DeviceResident`
  VRAM breakdown, wired into `profile=True`'s existing report
  (`resident_<name>_mb` alongside `tracked_footprint_mb`). Real gap
  found while landing this, not yet fixed: no shared `MemoryManager`
  reachable from the trainer node, so `memory_manager_stats` is always
  `None` in a real run today -- see the module docstring.

**Server / graph**
- `server/graph_executor.py` -- topological execution, port-compatibility
  checking, `ExecutionContext` threading; pre-existing, matches the
  design's construction-time model already.
- `server/nodegraph_registry.py` -- palette list; currently matches every
  concrete `Node` subclass in `nodes/`.

**Testing**
- 41 smoke tests under `nodes/smoke_tests/` (plus `manager/`, `server/`
  ones), runnable on CPU with no ComfyUI/XPU present --
  `nodes/smoke_tests/run_all.py` runs all of them.

## Still open, in priority order

See `docs/training_pipeline_design.md` section 10 for the full reasoning
behind this order.

1. Per-block profiling instrumentation, then `CheckpointPlacementPolicy`/
   `GreedyRatioPlacement` (2.3) -- blocked on the instrumentation, not
   the policy class
2. Wire `AdapterStrategy` into `ComfyUNetLoRANode`'s real construction
   path, then `DoRAAdapter` (3.1)
3. `NF4WeightStore` (3.3) -- single highest-value remaining item, needs
   a real dequantization implementation
4. Validation only, code already exists: `RescaledZeroTerminalSNRSchedule`
   end-to-end training run (1.4); `LoRAPlusGroups` actually tuned against
   a `UniformGroups` baseline (3.4)

**Not yet its own item, nothing above needs it yet:** thread a shared
`MemoryManager` through optimizer construction so `ResourceProfile`'s
`memory_manager_stats` is ever populated in a real run -- see
`nodes/memory/profile.py`'s module docstring.

**Not recommended near-term:** `ComponentRegistry`/`TrainingRecipe`/
`PipelineFactory` (5.3, 5.4) -- nothing built through
`nodes/components/` so far is graph-editor-selectable, so the problem
these solve hasn't materialized.

**Deferred or rejected**, reasoning in full in section 7:
`AutoResourcePolicy`, automatic eviction inside `MemoryManager`,
layer-wise base offload, flow matching, GaLore, 8-bit optimizer moments.

---
Last synced against `docs/training_pipeline_design.md` at commit
`157bc7a` (2026-08-15).
