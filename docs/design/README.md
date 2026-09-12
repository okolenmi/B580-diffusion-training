# Design docs index

*New here? Start at the root [`README.md`](../../README.md) for a map
of all the docs in this repo -- this folder is the deepest one, and
the design work with the most current activity
(`resources-controller/`) is nested inside it.*

This folder used to be one file, `docs/training_pipeline_design.md`
(~1,900 lines). It's been split into one file per top-level section so
each topic can be found, linked, and read on its own -- the content
itself is unchanged, only reorganized. **Section numbers are preserved
across the split** (a reference to "section 3.1" elsewhere in this repo
still means the same thing; use the table below to find which file it's
now in). Heading levels were promoted by one (the original `##` section
heading is now each file's `#` title) so each file reads correctly on
its own.

## What this design is

A from-scratch design for the training pipeline -- VRAM-savings-first,
strict OOP, real composition, no singletons -- developed independently
of `nodes/`'s actual classes, then compared against them once the
design was settled. Originally written in three passes (foundational
architecture, then a review pass that added techniques with real
published evidence behind them, then a fixes-only pass), merged into
one document instead of left spread across three (and now split back
into several, but by topic rather than by authoring pass). The process
that produced the original design is in git history, not repeated
here.

**Status update, 2026-08-12: the original numbered backlog (12 items)
is fully implemented and equivalence-tested.** Every item that the
backlog used to list -- `DiffusionProcess`/`DeviceContext`, the
`DeviceResident` ABC (with `OptimizerHandle`/`TrainableModel`/
`TextEncoder` all conforming to it), the `ParameterGroupPolicy` fix,
`LoRAScalingPolicy`, Min-SNR's v-prediction branch plus
`P2LossWeighting`, `ActivationCheckpointingStrategy`, `ProjectLayout`,
the `AdapterStrategy`/`FrozenWeightStore` seam,
`TrainingStepPipeline`/`StepPhase`, `PrefetchingBatchSource`, and
`ResourceCoordinator`/`OffloadOrchestrator` -- is real, tested code in
`nodes/` today, not illustrative Python. Sections describing something
now implemented keep their rationale but no longer repeat the
illustrative class code -- that code is stale next to the real, tested
version, so read the real file (pointed to inline in each section)
instead of trusting a copy here that can drift. Real code that cites a
section number (e.g. `nodes/model/adapter_strategy.py` citing 3.1) is
citing the section for its *rationale*, which is why that rationale is
kept even where the illustrative code it originally sat next to has
been removed.

What's left with illustrative code, in full, is only what's genuinely
still open: `NF4WeightStore`'s real-run quality check (the forward path
itself is implemented, see 3.3), and two pieces of *validation* work
that landing code alone can't finish (`RescaledZeroTerminalSNRSchedule`
needs a real end-to-end v-prediction training run; `LoRAPlusGroups`
needs a real tuned run, not just existing as an opt-in policy).

## Reading order

Don't read top-to-bottom unless you actually want the full design
history. For "what's real right now," start at file 08 (Implementation
status). For "what's planned next," file 09 (Backlog).

| # | File | What's in it |
|---|---|---|
| -- | [`01-design-goals-and-constraints.md`](01-design-goals-and-constraints.md) | The 7 constraints every design choice here is checked against (VRAM-first, strict OOP, no singletons, composition over inheritance, etc.). Read this once, refer back rather than re-reading. |
| 1 | [`02-foundational-ontology.md`](02-foundational-ontology.md) | Base vocabulary: `Builder` vs runtime object, `DeviceResident`, pooled buffers, `NoiseSchedule`/`DiffusionProcess`, device backend as Strategy, config as an injected value object. |
| 2 | [`03-training-step-orchestration.md`](03-training-step-orchestration.md) | The step loop as a pipeline of phases, resource policy, activation checkpointing, text encoder caching, batch prefetching. |
| 3, 4 | [`04-lora-adapter-mechanics-and-loss-weighting.md`](04-lora-adapter-mechanics-and-loss-weighting.md) | `AdapterStrategy` (plain LoRA vs DoRA), `LoRAScalingPolicy`, `FrozenWeightStore` (incl. NF4), per-parameter-group learning rates; Min-SNR/P2 loss weighting. |
| 5 | [`05-coordination-registry-observability.md`](05-coordination-registry-observability.md) | `ResourceCoordinator`, offload ordering, `ResourceProfile`, why `ComponentRegistry`/`TrainingRecipe` are deliberately not built yet. |
| 6 | [`06-composition-walkthrough.md`](06-composition-walkthrough.md) | One concrete LoRA run traced through every piece above, to make sections 1-5 legible as a whole. |
| 7 | [`07-deferred-or-rejected.md`](07-deferred-or-rejected.md) | Considered and left out on purpose, with real reasoning: `AutoResourcePolicy`, automatic eviction, layer-wise base offload, flow matching, GaLore, 8-bit optimizer moments. |
| 8, 9 | [`08-validation-and-implementation-status.md`](08-validation-and-implementation-status.md) | **The most reliable "what's actually real" table in this repo.** What's implemented, what's partial/unvalidated, what's explicitly out of scope. |
| 10 | [`09-prioritized-backlog.md`](09-prioritized-backlog.md) | What's left, in order, with reasoning for the order. |
| 11 | [`10-node-surface-and-precision-control.md`](10-node-surface-and-precision-control.md) | Planning for node-graph surface/precision control -- mostly superseded by, and cross-referenced from, `resources-controller/`. |

## The active follow-on work

[`resources-controller/`](resources-controller/README.md) -- the
Resources Controller & precision redesign. This is where new design
work has actually been landing recently; it's the single most current
doc in this repo as of the last docs pass. Start there for anything
resource-policy or precision related, not in the files above.
