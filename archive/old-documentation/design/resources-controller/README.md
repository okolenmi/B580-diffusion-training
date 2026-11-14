# Resources Controller and precision

The redesign that made resource control an explicit, checkable part of a
graph rather than a side effect: a Resources Controller node that emits a
verified resource pack, a LoRA-injection node that consumes it, and a
trainer whose residency is deterministic per phase instead of reactive.

**These phases are built.** What is kept here is the design reasoning —
what was decided, why, and what was rejected — because that is what the
code's comments cannot carry. For the shipped behaviour, read the code.

## Priority

Where this folder and `docs/design/` disagree, **this folder wins**: it
reflects the more recently decided direction. `docs/design/` remains the
record of what shipped and why for everything outside this redesign, and
`08-consolidation.md` lists the items where the two were reconciled. It is
not frozen — it is simply not the tie-breaker for anything this folder
touches.

## Reading order

| File | What it holds |
|---|---|
| [`01-context-and-ground-truth.md`](01-context-and-ground-truth.md) | Why the redesign exists, and the measured ground facts it is built against rather than assumed ones. Includes the composition-mechanism question that had blocked Phase 4, and its resolution. |
| [`03-phase-3-interactive-node-support.md`](03-phase-3-interactive-node-support.md) | Editor, `core.py` and introspection support for interactive nodes. |
| [`04-phase-4-resource-preset-abstraction.md`](04-phase-4-resource-preset-abstraction.md) | The `ResourcePreset` abstraction and its construction mechanics. |
| [`05-phase-5-resources-controller-node.md`](05-phase-5-resources-controller-node.md) | The Resources Controller node. Its scope was narrowed: it produces a verified `LoRATrainingResources` pack and does **not** inject LoRA itself, which is Phase 6's job. |
| [`06-phase-6-lora-training-config.md`](06-phase-6-lora-training-config.md) | `LoRATrainingConfigNode` — consumes the pack and injects LoRA, including locking rank when continuing from an existing adapter. Its original sketch of the `TrainerNode` integration turned out wrong and moved to 09. |
| [`08-consolidation.md`](08-consolidation.md) | Low-synergy items resolved against the main design docs instead of being left to drift. |
| [`09-trainer-integration-and-vram-safety.md`](09-trainer-integration-and-vram-safety.md) | `ManagedLoRATrainerNode` — its own step loop, taking `trainer` directly rather than adapting into the main route's ports — plus the `strict` VRAM-budget mode that makes the ceiling a guarantee rather than best-effort. |

Two files that used to sit here are gone.
`02-phase-1-and-2.md` covered the lazy-resource work — header-only
checkpoint inspection and `ModelWeights` loading a checkpoint on first
access rather than eagerly. That shipped: it is
`nodes/model/resource_inspection.py`, pinned by
`smoke_test_resource_inspection.py`, and the doc restated it.
`07-post-phase-6-bugfixes.md` was a record of four bugs found by using the
editor after Phase 6 landed; all four were fixed and live in their code and
tests, and the file was a low-value historical record.