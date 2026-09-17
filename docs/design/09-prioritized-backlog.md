*[← docs/design index](README.md)*

# 10. Prioritized backlog

**The original 12-item backlog is entirely complete, and so are the two
items that led this list after it** -- `DiffusionProcess`/`DeviceContext`,
`DeviceResident` (with `OptimizerHandle`/`TrainableModel`/`TextEncoder`
all conforming), the `ParameterGroupPolicy` fix, `LoRAScalingPolicy`,
Min-SNR's v-prediction branch + `P2LossWeighting`,
`ActivationCheckpointingStrategy`, `ProjectLayout`, the
`AdapterStrategy`/`FrozenWeightStore` seam (now live-wired into
`ComfyUNetLoRANode`'s real construction path -- see 3.1),
`TrainingStepPipeline`/`StepPhase`, `PrefetchingBatchSource`,
`ResourceCoordinator`/`OffloadOrchestrator`, `ResourceBudget`/
`ResourcePolicy`/`ManualResourcePolicy` (the latter two later removed --
see 2.2), `ResourceProfile`,
`BlockCost`/`CheckpointPlacementPolicy`/`EveryBlockPlacement`/
`GreedyRatioPlacement`/`BlockProfileCollector`/`ProfilingCheckpointing`,
`DoRAAdapter`, `NF4WeightStore`'s quantization itself, and now
`NF4WeightStore`'s forward-path wiring (`NF4LoRALinear`/`NF4LoRAConv2d`,
a real `frozen_weight_store` port on `ComfyUNetLoRANode`) are all real,
tested code -- see section 9.1 for exactly where each one lives, and 2.2
for the real deviations `ResourcePolicy`'s implementation took from this
document's own illustration once it was actually built, and for why it
was removed again afterward. What follows is
a fresh list: only what's actually still open, ordered by what unblocks
what, sized to be independently landable slices, each one
equivalence-tested against whatever it replaces (or, for the
validation-only items at the end, tested by a real training run instead)
before anything switches over to it.

1. **Verify `NF4WeightStore`'s quality against a real training run**
   (3.3). The forward path is wired and equivalence-tested against a
   fixed dequantized reference, but the diffusion-specific quality
   question -- does NF4's real ~9% relative RMSE (see that module's own
   docstring) actually produce usable LoRA training results on this
   project's real UNet -- still needs checking directly, not assumed
   from QLoRA's own LLM benchmarks. A `MemoryManager`-backed scratch
   buffer for the dequantized tensor (real, separate optimization, not
   blocking correctness) is the other remaining piece -- see
   `nf4_lora_layer.py`'s own docstring. `DoRAAdapter` doesn't honor
   `NF4WeightStore` yet (`QDoRA`) -- real, separate follow-up, not
   done here.

**A real gap surfaced while landing `ResourceProfile`, not yet its own
backlog item because nothing above needs it yet:** there is no single
shared `MemoryManager` instance reachable from
`SupervisedLoRATrainerNode.build()` -- each optimizer execution strategy
that uses one (`ChunkedScratchBufferStrategy`) constructs its own private
instance when none is injected, and no `Composed*OptimizerNode` exposes
a port to inject a shared one. Harmless today (each strategy's own
private manager is internally consistent), but it means
`ResourceProfile.memory_manager_stats` is `None` in every real run, and
it'll become a real problem the moment two things need to share one VRAM
budget on purpose (e.g. a future `AutoResourcePolicy`, or several
strategies deliberately pooling scratch space) -- worth threading a
shared `MemoryManager` through optimizer construction as part of
whichever future item first has a concrete reason to, rather than
speculatively now.

**Validation work, not construction -- real code exists for both, what's
missing is a real run:**

- **`RescaledZeroTerminalSNRSchedule` + `VPredParameterization`, end to
  end** (1.4). The classes are implemented and `DiffusionProcess` already
  rejects the unsound eps-prediction pairing at construction. What's
  missing is a real training run and qualitative image-quality evaluation
  -- does it actually fix medium-brightness clustering on this project's
  own data. Unlike everything numbered above, there's no old code path to
  equivalence-test against; this needs real training runs to trust, not a
  unit test.
- **`LoRAPlusGroups`, actually run** (3.4). Opting in is now a one-line
  `group_policy=LoRAPlusGroups(...)` change on any `Composed*OptimizerNode`
  (2.2) -- what's missing is running it on a real LoRA training job and
  comparing against a `UniformGroups` baseline, at whatever `ratio` (the
  `16.0` default is an unverified starting point) turns out to matter for
  this project's own data.
- **`GreedyRatioPlacement`, wired and run** (2.3). `BlockProfileCollector`/
  `ProfilingCheckpointing` produce real `BlockCost` numbers now, but
  nothing has actually run them against this project's own UNet yet, and
  `ComfyUNetLoRANode` has no port to select `ProfilingCheckpointing` or
  `GreedyRatioPlacement` in the first place -- both real, separate,
  smaller follow-ups once a first profiled run's numbers exist to wire
  a real placement decision against.
- **`DoRAAdapter`, a real training run** (3.1). Equivalence-tested
  against an independent reference implementation and identity-at-init,
  but not yet run on this project's own data to confirm the quality
  improvement DoRA reports in its own published benchmarks (LLaMA/LLaVA/
  VL-BART, not diffusion UNets) actually shows up here too. Checkpoint
  save/load (direction + `.dora_scale` magnitude + alpha) is real now
  for the common, unsplit case -- see 9.1/9.2 -- so this item is
  validation-only, same as the others in this list.
- **A tiny-parameter (`< 10,000` element) `ExecutionStrategy` for
  Adafactor's cross-parameter batching case** (11.1). Found while
  retiring the redundant legacy Adafactor nodes: `ChunkedXPUAdafactor`
  ties every tiny parameter in the whole optimizer together into one
  shared clip/EMA state, which is a batching-strategy concern, not a
  per-parameter algorithm one -- `AdafactorAlgorithm` has no way to see
  other parameters in the same optimizer by design (see
  `algorithms/base.py`). Would need something like
  `ShapeGroupedBatchStrategy`, but grouping "every parameter under a
  size threshold, any shape" instead of "same shape" -- real, separate
  feature work, not a formula fix. `AdafactorOptimizerNode` stays
  registered until this lands (see `docs/known-issues/open.md` and
  `nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py` for
  the confirmed, measured gap this would close).
- **`core`/`manager` coupling in the model and dataset domains** (new
  section, not yet numbered above -- see `docs/architecture.md`). The
  `optimizer/` domain's `Algorithm`/`ExecutionStrategy` split proved a
  domain can be fully separated from `core/`'s legacy implementation,
  verified equivalent, and the old wrapper retired -- `nodes/model/`
  (LoRA/UNet injection: `core.lora`, `core.unet_wrapper`),
  `nodes/model/text_encoder.py` (`core.clip_encode`), and
  `nodes/dataset/managed.py` (`manager.loader`) haven't had that done at
  all yet: single implementation, still wrapping `core`/`manager`
  directly, no competing alternative to retire. Real future work, not
  cleanup debt -- sized much bigger than the optimizer domain was (UNet
  forward passes and LoRA injection are substantially more surface than
  three optimizer formulas), or bigger than dataset ingestion, and not
  scoped further here. Whoever picks this up should decide which
  sub-piece (model vs. dataset) goes first.

**Not recommended as near-term work, with reasoning kept where it's
argued in full:** `ComponentRegistry`/`TrainingRecipe`/`PipelineFactory`
(5.3, 5.4 -- nothing built through `nodes/components/` so far is
graph-editor-selectable, so the side-by-side-registration problem these
solve hasn't materialized). **Deliberately deferred or rejected, for the
reasons section 7 gives in full:** `AutoResourcePolicy`, automatic
eviction inside `MemoryManager`, layer-wise base offload, flow matching,
GaLore, 8-bit optimizer moments.
