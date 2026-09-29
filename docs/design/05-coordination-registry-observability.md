*[← docs/design index](README.md)*

# 5. Coordination, registry, and observability

## 5.1 `ResourceCoordinator`: a registry of `DeviceResident`s, offload ordering made explicit

**Implemented**, unchanged from the design: `ResourceCoordinator`
(`nodes/memory/coordinator.py`) -- tracks every `DeviceResident` a run
has constructed via explicit `register()` calls, never reflection, never
a global registry reached for by import; `total_footprint_bytes()` and
`offload_all_except()` cover the "what do I own, offload everything but
these" operations that were otherwise easy to get subtly wrong by hand.
Sequenced correctly, per the original plan: it landed only once
`OptimizerHandle`, `TrainableModel`, and `TextEncoder` were all real,
tested `DeviceResident`s to actually coordinate.

## 5.2 `OffloadOrchestrator`: event-driven, reusing the existing pub/sub shape

The project already had a working, correctly-designed pub/sub mechanism
for cross-cutting concerns: `MonitorBus`/`MonitorHandle`, explicitly
injected rather than a singleton, already documented as safe to call from
a worker thread. Rather than invent a second event system for offload
orchestration, this reused that same shape.

**Implemented**, unchanged from the design: `TrainingLifecycleEvent`
(marker base -- `CacheRebuildStarting`, `PreviewGenerationStarting`,
`CheckpointSaveStarting`) and `OffloadOrchestrator` (subscribes to
events, drives a `ResourceCoordinator` in response) -- both in
`nodes/memory/coordinator.py`.

**Still true, and worth restating exactly as before:** this is the
principled version of what `core/trainer.py`'s hand-written offload calls
still do today, ad hoc, per call site -- for `nodes/`, not a claim that
it retroactively fixes `core/trainer.py`. It's also explicitly not, by
itself, a fix for the still-open "device lost"/hang-after-VRAM-pressure
report in `docs/known-issues/open.md` -- that report's own leading
hypothesis is a missing explicit `synchronize()` on an async offload
path, a correctness bug this orchestrator's *existence* doesn't fix. It
fixes the *coordination* problem, which is necessary but not sufficient
-- see section 9.3.

## 5.3 `ComponentRegistry`: versioned, side-by-side registration

`server/nodegraph_registry.py` is already a plain name -> class registry,
which is fine for "the graph editor needs to resolve a class name." What's
missing, and what the project's own migration discipline actually needs,
is a way for a `nodes/components/`-style rewrite to be registered
*alongside* the legacy adapter it's replacing, both live, both usable, for
however long the equivalence-testing window takes -- exactly the pattern
`nodes/optimizer/`'s composed nodes vs. legacy-wrapping nodes already
follow *by convention*.

**Not implemented -- still not urgent, but for an updated reason.**
`nodes/components/` is no longer empty -- `diffusion.py`, `device.py`,
and `layout.py` are real, migrated content now (1.4-1.6) -- so the
original justification ("nothing has moved yet") is stale. The actual
current reason this still isn't needed: none of those three are
registered as selectable graph-editor `Node` types at all -- they're
plain constructed objects wired in through ports and constructor
arguments (`diffusion_process` on `SupervisedLoRATrainerNode`,
`project_layout` on the four Nodes that need it), not competing
named implementations a picker has to choose between. `ComponentRegistry`
solves "the graph editor needs to offer X-old and X-new side by side";
nothing built through `nodes/components/` so far needs that, since
nothing built through it is graph-editor-selectable in the first place.
Worth re-examining if a future `nodes/components/` migration *is* exposed
as a selectable `Node` (a `DiffusionProcess`-choosing Node in the graph
editor, say) -- listed for completeness, cheap to add exactly when that
happens.

## 5.4 `TrainingRecipe` / `PipelineFactory`: declarative composition

A value object describing a full run (dataset config, model config,
optimizer config, schedule, resource budget) plus a factory that turns it
into wired, constructed runtime objects -- the Abstract Factory pattern
applied to "build me a whole pipeline," rather than wiring every
`Builder` by hand each time. Not proposed as a replacement for the graph
editor (which has real value as an interactive, inspectable construction
UI) -- a second, programmatic entry point for the exact same underlying
`Builder`/runtime-object model, useful for tests, scripts, and
(longer-term, speculative) as a possible bridge toward driving `nodes/`
from a config file the way `core/` already is, without `nodes/` and
`core/` sharing any code to do it. **Not implemented -- still not
recommended as near-term work**, for the same reason as before: nothing
currently needs it (there's no test harness or script today that
constructs a full `nodes/` pipeline programmatically instead of through
the graph editor), and it depends on `ComponentRegistry` (5.3) existing
first.

```python
@dataclass(frozen=True)
class TrainingRecipe:
    dataset: dict
    model: dict
    optimizer: dict
    schedule: dict
    budget: ResourceBudget


class PipelineFactory:
    def __init__(self, registry: ComponentRegistry):
        self._registry = registry

    def build(self, recipe: TrainingRecipe) -> "TrainingStepPipeline":
        ...  # resolves each section against the registry, constructs,
             # wires Builders in dependency order -- the programmatic
             # equivalent of what GraphExecutor already does for a
             # graph-editor-submitted graph
```

## 5.5 `ResourceProfile`: one aggregate VRAM report

**Not implemented -- still open, and cheaper to build now than when
this was written**, since every `DeviceResident` it would aggregate
(`OptimizerHandle`, `TrainableModel`, `TextEncoder`) is now real:

```python
@dataclass(frozen=True)
class ResourceProfile:
    per_resident_bytes: dict[str, int]
    memory_manager_stats: dict[str, Any]
    allocator_stats: dict[str, float] | None  # from DeviceContext.memory_stats()

    @classmethod
    def capture(cls, coordinator: ResourceCoordinator, memory: "MemoryManager",
                device_ctx: DeviceContext) -> "ResourceProfile":
        return cls(
            per_resident_bytes={name: r.footprint_bytes()
                                 for name, r in coordinator._residents.items()},
            memory_manager_stats=memory.stats(),
            allocator_stats=device_ctx.memory_stats(),
        )
```

Directly generalizes what `profile=True` already reports (allocated/
reserved MB) into a per-component breakdown -- "how much of my VRAM is
the text encoder cache vs. optimizer scratch vs. the model itself," which
the current single allocator-level number can't answer. Real, standing
diagnostic value for the still-open VRAM-pressure investigation in
`docs/known-issues/open.md` -- see the backlog,
`docs/design/09-prioritized-backlog.md` section 10.

## 5.6 Concurrency contract, stated explicitly

Stated once, precisely, rather than left implicit (which is how a
`PrefetchingBatchSource` worker thread, 2.5, could otherwise become a
real race someone finds the hard way) -- **this is no longer aspirational:
`nodes/dataset/prefetch.py` cites this section directly as the contract
its worker thread has to honor**:

- **Single-threaded by default.** `StepPhase.run()`, `DeviceResident`
  methods, `MemoryManager` methods, `ResourceCoordinator`/
  `OffloadOrchestrator` methods: none of these are safe to call from more
  than one thread concurrently, and none of them need to be -- a training
  run has exactly one thread driving the step loop.
- **Explicitly cross-thread-safe, by design, documented as such at the
  point of use:** `MonitorHandle.report()` (already true and already
  documented -- called from a FastAPI worker thread today);
  `ExecutionContext`'s cancel signal (already a `threading.Event` for
  exactly this reason); `PrefetchingBatchSource`'s internal queue (its
  *only* job is being a safe hand-off point between its worker thread and
  the training thread -- `queue.Queue` already gives this for free, so
  this isn't new design work, just a contract worth stating).
- Nothing else should grow a background thread without updating this
  list and justifying it the same way.

## 5.7 The Acyclic Domain Dependency Rule

The project's practiced discipline (`optimizer/` doesn't import `model/`;
domain ABCs live in each domain's own `handle.py`), including in
`nodes/components/` now that it's real, not just planned:

> A domain package (`dataset/`, `model/`, `optimizer/`, `train/`,
> `monitor/`, `components/`) may depend downward on `core.py` and
> `memory/` (and, for now, read-only on `core/`/`manager/` per the
> existing wrap-don't-copy rule) -- never sideways on another domain
> package's *implementation*. Cross-domain references go through that
> domain's `handle.py` ABCs only (e.g. `train/` depends on
> `model.handle.TrainableModel`, never on `model.lora_injector`
> directly). A dependency that seems to need to go sideways is a signal
> the shared piece belongs in `core.py`, `memory/`, or a new
> domain-independent module -- not that the rule should bend.

## 5.8 The Training Monitor dashboard

One graph node (`TrainingProgressMonitorNode`) + one bus
(`MonitorBus`, section 5.2's shape) feed one page
(`server/static/monitor_dashboard.{html,js}` via `routes_monitor.py`).
There is deliberately no per-metric monitor node: a run already reports
everything through one `MonitorHandle`, so splitting the *screen* into
more graph nodes would add wiring without adding information.

**Message vocabulary** on the bus (checked by the dashboard in this
order, before the step-shaped filter -- `run_end` carries a `step` but
is not a step report):

- `{"type": "connected"}` -- sent by the SSE route on subscribe.
- `{"type": "clear"}` -- a fresh run started against this `monitor_id`
  (`MonitorBus.clear()`, called from `build()`); the dashboard resets
  charts/stats/CSV history so two runs never overlay silently.
- `{"type": "run_end", "step", "cancelled", "t"}` -- emitted by **both**
  trainer routes at **both** exits (normal and cooperative cancel).
  Without it an open SSE stream looks identical during "running" and
  "finished", so "is it done or hung?" was unanswerable.
- Step-shaped reports: `{step, total_steps, loss, lr, t, ...}` plus the
  optional keys below.

**Report keys the dashboard charts** (each absent -- never fabricated 0
-- when its source doesn't have it; charts draw gaps for missing series,
the same rule `t_bucket_losses` follows):

- loss + `loss_t_low/mid/high` -- the multi-series loss chart (colored
  per t bucket; the t-high divergence reads first as "destructive at
  strength 1.0").
- `vram_reserved_mb`/`vram_allocated_mb`/`vram_peak_reserved_mb` (both
  routes, `vram_{k}` naming shared with the main route's full
  `memory_stats()` dump) and `resident_{model,optimizer,text_encoder}_mb`
  -- the VRAM/residency chart; its card stays hidden until a report
  actually carries one of these keys.
- `vram_budget_mb` -- the handle's `usable_budget_mb()`, drawn as a
  dashed reference line at its true position (the chart's range joins
  reference values so an out-of-span ceiling doesn't clamp to the plot
  edge and read as "equal to the largest sample").
- `{label}_ms` + `step_total_ms` -- the phase-timing chart. Labels come
  from the shared `_phase_label` (`step_pipeline.py`), so the managed
  route's `TRAIN_STEP_TIMING=1` report and the main route's `profile`
  report use identical key names. Series are discovered from the first
  timing-bearing report (phase lists aren't known client-side).
- `grad_norm` -- only when gradient clipping is on: the clip already
  measured the pre-clip total norm, so the report key is free. Absent
  means "not measured" (measuring separately would cost a grad pass +
  device sync every step), never a placeholder.

**Screen features**: composition instead of a flat grid of equal
cards -- a hero status block (step, loss, smoothed and ETA as large
readouts, a full-width progress bar, then a meta row with steps/sec,
elapsed time and the optimizer id the report carried); the loss chart
beside a run-stats rail (best loss, LR, the grad-norm row, per-t
bucket rows showing each range's latest *measured* loss -- the
current report's own value when its window sampled that range,
otherwise the last measured one, dimmed and tagged `@step` so a
carried number never claims to be this step's measurement (at batch
2 a bucket is often unsampled; the chart still draws per-step gaps --
the chart answers "what happened at step N", the rail "where does
each range stand now"), bar length comparing the three displayed
magnitudes on their shared scale; a range never measured this run
keeps the em dash and an empty bar; tooltips state the provenance
step either way, and a memory block with a
budget-utilization meter -- color-banded by pressure: accent below
90%, amber 90-99%, red at/over the stated budget -- plus the session
peak); and the VRAM/residency and phase-timing charts side by side
below, each hidden until a report carries their keys (an all-hidden
row collapses to zero height rather than leaving a gap). Blocks whose
keys depend on run configuration -- the memory section, the grad-norm
row -- appear only once a report actually carries them and are hidden
again by `clear`, so a CPU run never shows permanently-empty sections.
Graph controls drive all three charts (shared step domain) and sit with
the chart itself: Freeze/Resume and the visible-items count
(50/100/250/1000/All, default All) are centered in the loss card's
header row, and a horizontal slider under the chart pans back through
recorded history -- dragging it to the far end returns to live; it is
disabled under All, where there is nothing to pan. Freeze/Resume locks
the window where it is while reports keep arriving (the readout keeps
showing `/ latest` so a held view still reports that newer data exists).
While the pointer holds the slider thumb, its `max`/`value` are frozen:
every report would otherwise rewrite them under the finger, and the
browser's drag tracking then fights those writes -- the thumb teleports
(worst at the live end, where the index grows per report); the far-end
test meanwhile uses the frozen max, so a drag that reaches the right
edge stays live, and release resyncs the geometry to current data. The
slider's track has no accent-filled segment (a left-to-thumb fill reads
as progress, but a window has no completion semantics) -- neutral track,
accent thumb only. The window is a `LossChart` view range:
out-of-window points are clipped at the plot border and the y-range and
tooltip only consider in-window points, so an old spike can't keep
compressing a panned-back view. The charts keep up to 100k records
(the default 1000-point cap would silently truncate exactly the oldest
records the slider exists to reach).
Legend click hides/shows a series (hidden series leave the tooltip and
the axis range, and show struck-through in the legend itself), and
session CSV export (`Export CSV`: one row per step report this browser
session received, columns = union of report keys in first-seen order,
missing cells empty -- session-scoped because that's all the page ever
had; the bus keeps only its 500-event history).

---
