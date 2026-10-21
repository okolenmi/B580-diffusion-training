# 10. Events: a second plane beside the data wires

Status: **proposal; the owner called events "the biggest helpful update".**
Everything else in `rework/` leans on this plan (dataset prompt swap, prewarm,
periodic saving, probes, widgets, templates). Inputs:
`audit/backend-graph-engine.md`, `audit/frontend-graph-ui.md`.

> **Superseded in part (2026-10-13 18:21):** the owner dropped all compatibility and migration work. Read `90-build-plan.md`: no shims, aliases, deprecation windows, old-port support or old-versus-new parity gates; the new tree is built clean beside `nodes/`.

## 1. What the owner asked for

The graph only moves forward; sometimes it needs to go back. Example given:
**after each epoch, a node that swaps tags activates again** ("one tag per image"
experiment). The latent-prompt controller signals the end of the epoch when the
last element is consumed or cached. A dataset that changes a lot may announce
"prompts changed". Template presets "simply can be subscribers of specific graph
events and update info in their widgets".

## 2. Where we start from

* The run is a **forward-only DAG**, each node's `build()` called once in
  topological order; the trainer's `build()` *is* the loop
  (`runtime.py:379-417`). A node instance is created per node and not
  retained by the runtime (`cls(node_context).build(**inputs)`).
* Events exist only **from the run to observers**: lifecycle events on the
  numbered, replayable `EventBus` (SSE `/events`), per-node report frames on
  the `MonitorBus` (`node:<id>`, `graph:run`), the child's event file, the
  `Completion` side channel, the cancel flag. **No node can subscribe to
  anything, and no node can signal another during a run.**
* The epoch boundary is already a clean seam. `FetchBatchPhase` catches
  `StopIteration` and calls `iter(batches)` again; every
  `TrainingBatchSource.__iter__` is "a fresh pass each time"; the prefetcher
  starts a **new worker thread per `__iter__`** (`prefetch.py`). So nothing of
  epoch N+1 is read before the consumer asks for the next iterator.

## 3. The idea: two planes

| | Data plane (today) | Event plane (new) |
|---|---|---|
| Carries | typed values on wires | named signals with a small payload |
| Shape | DAG, evaluated once, topological | any directed graph; cycles allowed |
| Direction | forward only | any direction |
| When | at build, one time | any time during the run, at safe points |
| In the editor | solid wires, sockets on the sides | dashed wires, event ports in a strip on the node |

The event plane never changes *what a value is*. A signal says "something
happened"; state changes go through a **shared handle** that already travels
on the data plane (for example a `PromptTable`, plan 50), so payloads stay
small and ordering stays obvious.

## 4. The model

```python
@dataclass(frozen=True)
class EventType:                 # a class, so wires can be type-checked by MRO
    def summary(self) -> dict: ...         # small, JSON-safe, used by UI mirroring

class EpochEnding(EventType):    epoch: int; samples: int
class EpochStarted(EventType):   epoch: int
class PromptsChanged(EventType): version: int; changed: int; of: int
class StepFinished(EventType):   step: int; total: int; metrics: dict
class CheckpointDue(EventType):  step: int; reason: str
class StopRequested(EventType):  reason: str
```

On a node:

```python
class SwapTagsNode(Node):
    EVENTS_IN  = {"epoch_ending": EventIn(EpochEnding, handler="on_epoch_ending")}
    EVENTS_OUT = {"prompts_changed": EventOut(PromptsChanged)}
    def on_epoch_ending(self, ev: EpochEnding, ctx: EventContext) -> None:
        self.table.pick_one_tag_per_image(seed=ev.epoch)
        ctx.emit("prompts_changed", PromptsChanged(self.table.version, ...))
```

* An **event port** is a `Port` with `kind="event"` and the payload class as
  its type. Compatibility reuses the existing MRO check.
* **Explicit wires** (decision E1): an event-out connects to event-ins. Many
  outputs may feed one input, and one output may feed many inputs (unlike data
  ports, there is no "one wire per input" rule on the event plane).
* The graph JSON gets a separate `event_edges` list. The data `edges` list and
  its validator codes stay exactly as they are.
* The engine also emits **graph events** with no wires needed because they are
  *about* the run: `run_started`, `node_state`, `node_progress`, `run_finished`,
  `run_failed`, `run_stopped`. The UI subscribes to those by topic; nodes only
  receive what is wired to them.

## 5. Runtime semantics (the rules that make it safe)

1. **Reactive nodes are retained.** A node that declares `EVENTS_IN` is kept
   alive by the runtime until the run ends; after `build()` the runtime
   registers its handlers with the `EventHub`. (Today the instance would be
   dropped; only the handles it returned live on.)
2. **One in-process hub per run**, in the same process as the nodes (the run
   already lives in a child process, ADR 0002). It lives on `ExecutionContext`
   (`ctx.events`), beside `monitor_bus`, `cancel_event`, `memory`.
3. **Dispatch is synchronous**: `emit()` calls every subscriber's handler, in
   a fixed order (topological order of the subscribing nodes), in the
   emitter's thread, and returns when all have finished. This is what makes the
   epoch boundary a **barrier**: the loader emits `EpochEnding`, the tag
   swapper rewrites the prompts, the handler returns, only then does the
   loader start the next pass.
4. **Emit only at safe points the emitter defines.** The emitter chooses where:
   a batch source emits between passes, the trainer between steps. A handler
   may assume the emitter is paused.
5. **Only the run thread emits.** Worker threads (the prefetcher) must not call
   `emit`; they may `hub.post()`, which queues for the run thread to drain at
   its next safe point (the trainer drains once per step).
6. **Re-entrancy is bounded.** A handler may emit; nesting depth is capped
   (default 8) and a per-run counter per event wire stops runaway loops with
   an `EventLoopError` that names the wire. Cycles on the event plane are a
   validator **warning** (`event_cycle`), not an error.
7. **Handler failure fails the run, loudly**, attributed to the node
   (decision E5). A silent skip would hide an inconsistent dataset.
8. **Not durable, not replayed to nodes.** Events are not persisted. Handlers
   must be idempotent in the sense that matters: payloads carry indices
   (`epoch`, `version`), and node state derives from them.
9. **After a stop**, dispatch ends except `run_stopped`; the existing
   `Completion` rules for savers are unchanged.
10. **Events never reorder the data plane.** The topological order of `build()`
    is untouched; events only act after the nodes involved have been built.

## 6. Mirroring to the UI (and templates)

The hub forwards every event's `summary()` to the **existing** transports, so no
new channel is needed:

* to the `MonitorBus` stream `node:<emitter id>` as a frame
  `{kind: "event", name, summary, t}` (the child already forwards monitor
  frames to the server; this reuses that path);
* lifecycle-class events (those the UI must never miss) also as domain events
  on the numbered `EventBus`, so a reconnecting tab gets them replayed under the
  existing contract (`docs/design/backend/09-event-contract.md`);
* node state (`resolving`, `running`, `ready`, `error`) is the same mechanism:
  a `node_state` event, which is how plan 30's "processing... please wait"
  works for slow server-side steps.

Consequences for the Templates tab (plan 60): a block can bind to an **event
topic** the same way it binds to a display module today, so "a preset is a
subscriber that updates a widget" needs no special-case; the closed set of block
kinds gains `event-feed` and `status`.

## 7. Worked examples

**A. One tag per image, rotated each epoch**

```
Dataset ──data──► ... ──► Trainer
   │ (PromptTable handle, shared)
   └─ EpochEnding ──event──► Swap Tags ── PromptsChanged ──event──► Conditioner (invalidate + warm changed)
```

Sequence: the batch source finishes pass N, emits `EpochEnding(N)`; Swap Tags
mutates the shared `PromptTable` and emits `PromptsChanged(version, changed)`;
the conditioner warms exactly the changed prompts (and drops stale cache keys)
before returning; only then the source starts pass N+1 with the new prompts.

**B. Incremental prewarm** replaces the whole-dataset "universe" computation: the
conditioner reacts to `PromptsChanged` and to `RunStarted`.

**C. Save every N steps** without a `save_every_n_steps` port on the trainer:
`StepFinished` -> `Every N Steps(n=500)` -> `CheckpointDue` -> `Save Adapter`
(event-in `save`). The saver already knows how to handle a partial result.

**D. Preview render / evaluation probe** every N steps: the same trigger wired to
a probe node's `run` event-in.

**E. Early stop**: `StepFinished` (metrics) -> `Loss Plateau` -> `StopRequested`
-> the runtime sets the cancel flag, reusing the Completion path so the saver still runs.

**F. Memory pressure**: the runtime emits `MemoryPressure` from its admission
checks; a node may react (shrink a cache) instead of the trainer owning every
reaction.

## 8. Nodes this adds (small, generic)

`Every N` (counts `StepFinished` or `EpochEnding`, emits a trigger),
`Event Log` (a display node: last events on the canvas),
`Debounce`, `Counter`. Domain nodes (swap tags, saver, probes) declare their own
ports. No node here knows about any other node.

## 9. What a node author must follow (goes into plan 80)

* Declare event ports; do not reach into the hub by topic string.
* A handler is short, deterministic and non-blocking; no sleeps, no network.
* Payloads are frozen dataclasses with a `summary()`; heavy objects travel in
  shared handles, never in the payload.
* Never emit from a worker thread; use `post()`.
* Emit past events by index, not by object identity, so resume logic works.

## 10. Alternatives considered

| Option | Why not |
|---|---|
| Global pub/sub by topic string, no wires | Fast to build, but dependencies are invisible in the editor, and typing and validation disappear |
| Reactive dataflow (re-evaluate downstream when an input changes, as ComfyUI re-executes) | Would re-run expensive nodes (loading, encoding); our nodes are not cheap functions |
| Callbacks inside the trainer only (status quo) | Not expressible in the graph; each new reaction edits the trainer |
| `asyncio` event loop | Training is blocking device code in a thread; async buys nothing and complicates cancel |
| Make the graph cyclic | Breaks topological build and validation for little gain; events give the loop without it |

## 11. Validator and API changes

* New issue codes: `unknown_event_port`, `event_type_mismatch`, `event_cycle`
  (warning), `event_edge_unknown_node`.
* Catalog: `NodeInfo.events_in`, `events_out` (name, payload type, MRO list).
* Run payload: `event_edges: [{from_node, from_port, to_node, to_port}]`.
* The child event file gains `event` records (summary only); the supervisor
  forwards them to the monitor stream.
* Contract tests for the new frame, in the style of `test_event_contract.py`
  (the frontend may only read fields that exist).

## 12. Risks

* **Hidden coupling through shared handles** (a `PromptTable` mutated from a
  handler). Mitigation: shared handles expose a narrow, versioned API; every
  mutation bumps a version that appears in the event.
* **Handler cost on the step path.** Emits are per epoch or per N steps, not per
  op; `StepFinished` per step costs one dispatch with usually no subscribers
  (measure; skip the call when the topic has no subscriber).
* **A node that emits before its handlers exist** (events during `build()`).
  Mitigation: the hub queues events until all reactive nodes have registered,
  then flushes in order.
* **Debuggability**: add an event trace panel and a `--trace-events` log level.

## 13. Decisions

| # | Decision | Options | Recommendation |
|---|---|---|---|
| E1 | Subscription | explicit event wires; topic filters; both | Explicit wires for node-to-node; topics only for UI observers and engine events |
| E2 | Dispatch | synchronous at safe points; queued/async | Synchronous barrier semantics, plus `post()` for threads |
| E3 | Node lifetime | rebuild per event; retain instance | Retain reactive nodes for the run |
| E4 | Editor placement | event ports on the sides with data ports; strip on top/bottom | A separate strip (top = in, bottom = out) with dashed wires, so the data layout is untouched |
| E5 | Handler error | fail run; log and continue | Fail the run, attributed to the node; per-wire opt-out later if needed |
| E6 | Naming | free strings; typed classes | Typed payload classes; the wire type is the contract |
| E7 | Graph JSON | mix into `edges`; separate `event_edges` | Separate list (no change to existing validator and saved graphs) |

## 14. Open questions for the owner

1. Are explicit dashed event wires what you picture, or do you want events to
   be addressable by name without wiring (decision E1)?
2. For the tag-swap experiment: should the swap happen **before** the next
   epoch begins (my design: yes, barrier) or can it lag one epoch?
3. Should a failing handler stop the run (my default) or be skippable per wire?
