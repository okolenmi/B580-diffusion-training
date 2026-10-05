# Node-attached monitor widgets

*Design, not built. Recorded 2026-10-05 at the maintainer's request, before
any code, so the shape is argued on its merits rather than inherited from
whatever the first implementation happened to be.*

## The problem with the monitor as it stands

Monitoring today is one `MonitorNode` per stream:

```
TrainingProgressMonitorNode --(monitor_id)--> MonitorBus --> SSE --> dashboard
        ^
        |  optional "monitor" input
   ManagedLoRATrainerNode  reports({"step":..., "loss":..., "vram_reserved_mb":..., ...})
```

That shape has three costs, all of which show up as *waiting on the
codebase* rather than as a bug:

1. **The metric list is a compile-time constant in two places.** The
   trainer assembles one flat dict per step (`managed.py:944-986`:
   `loss`, bucket keys, `probe_*`, `weight_t_*`, `vram_*`, `resident_*`,
   `{phase}_ms`, …), and the dashboard separately knows which of those
   keys are series worth plotting. Adding a metric is not "add a number" —
   it is "add a number, then teach the frontend it exists", in a
   different language, and neither half fails when the other is missed.
   A key the dashboard does not know is silently dropped; a key the
   trainer stops sending leaves a gap nobody can distinguish from "no
   data this step".

2. **The node that has the data is not the node that exposes it.** The
   trainer computes everything; a separate monitor node carries the
   `monitor_id` and the trainer takes it as an *optional* input. So a
   graph that forgets the monitor node trains fine and reports nothing,
   and the failure is silence rather than a validation issue.

3. **There is no such thing as "just show me the loss chart".** A stream
   is an opaque blob per `monitor_id`. The dashboard's own layout is
   fixed: whatever it renders, it renders for the whole stream.

None of these are bugs. They are what a single-blob-per-stream monitor
is, and the question is whether that is the shape we want for the next
thing.

## The proposal

**A node may carry its own optional monitor widgets. The monitor window
is composed by the user out of whatever widgets the running graph
offers.**

- A **widget** is one named, self-describing stream with its own kind of
  presentation — a loss line, a VRAM/allocated area, a phase-timing bar,
  a scalar readout, a counter. It is a card, not a field.
- A node **declares** the widgets it can produce (zero or more). The
  trainer declares loss, grad-norm, per-phase timing, VRAM series, and
  whatever else it already computes; a future preview-generation node
  would declare "samples at this seed over time" without anyone editing
  the trainer.
- **After the graph is launched, every widget the running graph offers
  is visible in the monitor window** — that is the discovery step. The
  window answers "what can I show?" from the live run rather than from a
  hand-maintained list in the frontend.
- **The user builds the window** from that list: move widgets, resize
  them, expand one into a full card or collapse it to a sparkline, group
  and compose them.
- **Graph-scope widgets** (task start date, RAM/VRAM for the process,
  task queue position) are fed by graph-level data rather than by a
  node's step loop. This is the part that "will be implemented later",
  and it is separable: it is a second *source*, not a change to the
  widget model.
- **Layouts are presets**: a composed window can be saved and loaded,
  so a user who cares about loss + VRAM gets that window in one click.

The migration is mostly subtraction, which is the appealing part: most of
what `TrainingProgressMonitorNode` exists for becomes **a list of widgets
attached to the training node**, which already has the data and already
runs every step. The monitor node stops being a required participant and
becomes (at most) a compatibility shim for graphs that still wire one.

### Why this shape rather than the alternatives

- **Not "more fields on one monitor node."** That is what exists. The
  flat dict is the problem, not the solution to it.
- **Not "a widget per node, hardcoded to that node's class."** Then the
  dashboard has to know every node class, which is the same coupling
  moved rather than removed. The widget has to describe *itself* — its
  data shape and how to draw it — so adding one does not mean editing a
  registry elsewhere.
- **Not "the monitor decides what to show."** The user asked to compose
  the window. A system-chosen layout would be one more opinion the user
  cannot overrule, and would be wrong for everyone who wants a different
  subset.
- **Not one widget per step-value.** Step-rate data (loss, grad norm) and
  run-rate data (start time, queue position) have different natural
  cadences and different honest axes; making them one uniform thing
  either hides the distinction or over-generalises the widget interface.

## What it composes with

The memory rework landing alongside this is what makes the graph-scope
widgets possible rather than hypothetical: the ledger snapshot already
answers capacity, held and free, and MEM-07's preview answers what a run
would need before it starts. Those are exactly the "RAM/VRAM usage, task
start date" widgets, and they come from a source that exists.

`MonitorHandle`/`LiveMonitorHandle` and the `MonitorBus` stay the
transport. What changes is what flows over it: from *one blob per
monitor id* to *many self-describing widgets*, each with its own
identity. `monitor_id` survives as the widget's identity — it is already
"stable across runs", which is what a widget id needs to be.

## Open questions worth deciding before code

These are the ones where the design is genuinely unsettled, listed so the
next person does not have to rediscover them:

1. **How does a widget describe its own rendering?** A descriptor the
   backend sends (type + series keys + suggested axes), or a type name
   the frontend maps to a registered component? The second keeps the
   frontend in charge of presentation and the backend out of UI concerns;
   the first lets a new widget appear without a frontend deploy. The
   current dashboard's key→series knowledge is the thing that has to go,
   and this is the seam where it goes.
2. **What is a widget's identity across runs?** Per (node, widget
   name), or per node instance? Per-instance is more precise and makes
   "the loss chart for *this* trainer" unambiguous when a graph has two;
   per-declaration is more stable across edits to the graph.
3. **Sampling.** Loss is per step and can run to tens of thousands of
   points; VRAM and phase timings are per step but change slowly; start
   date is once per run. One transport, three cadences — downsampling
   has to be per widget, or the chart will render 40,000 points.
4. **What does a widget do when its node is not in the graph?** The
   window is built from what a *particular run* offers, so a saved
   layout naming a widget this graph cannot produce has to degrade
   visibly rather than silently disappear.
5. **Does the composition live per user, per graph, or per preset?**
   Presets are the ask; whether a preset is portable between graphs is
   the question that decides how much a widget has to carry about its
   own requirements.

None of these need an answer before the widget model itself can be
prototyped — items 1 and 2 are the ones that would be expensive to
change later, so they are the ones to settle first.

## Related

- [`05-coordination-registry-observability.md`](05-coordination-registry-observability.md)
  — why the monitor bus is injected rather than a global, which this
  keeps.
- [`backend/09-event-contract.md`](backend/09-event-contract.md) — the
  transport's own gap (no replay), which a widget stream inherits until
  it is fixed.