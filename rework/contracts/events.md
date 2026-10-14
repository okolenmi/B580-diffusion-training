# Contract: events

Implements plan 10. Python: `next/graph/events.py`.

## Plane

Data wires carry values forward once. **Event wires** (`event_edges`) carry small
typed notifications between nodes during a run, in any direction, and are handled
**synchronously**.

## Types

An event type is a frozen dataclass deriving `EventType` with `summary() -> dict`
(JSON-safe and small; heavy data travels in shared handles, the payload carries
indices and versions). Initial set: `RunStarted`, `StepFinished(step,total,metrics)`,
`EpochEnding(epoch,samples)`, `EpochStarted(epoch)`, `PromptsChanged(version,changed,of)`,
`CheckpointDue(step,reason)`, `StopRequested(reason)`, `MemoryPressure(level)`.

## Declaration

```python
EVENTS_IN  = {"epoch_ending": EventIn(EpochEnding, handler="on_epoch_ending", doc="")}
EVENTS_OUT = {"prompts_changed": EventOut(PromptsChanged, doc="")}
```

Compatibility of an event wire: the payload class of the output is a subclass of the
payload class of the input.

## Hub rules

1. One `EventHub` per run, on `ctx.events`, in the run's process and thread.
2. Reactive nodes (declaring `EVENTS_IN`) are **retained** until the run ends and their
   handlers are registered right after their `build()`.
3. `emit(node_uid, port, event)` calls every subscriber handler **synchronously**, in
   topological order of the subscribing nodes, and returns when all finished.
4. Only the run thread may emit. Other threads call `post(...)`; queued events are
   delivered when the run thread calls `drain()` at a safe point of its choosing.
5. Handler signature: `handler(self, event, ctx: EventContext) -> None`; `ctx.emit(port,
   event)` emits from the handler's own node.
6. Nesting depth is capped (8). A per-wire counter stops runaway loops with
   `EventLoopError` naming the wire.
7. A handler exception fails the run, attributed to the node (`EventHandlerError`).
8. No persistence, no replay to nodes.
9. After a stop, dispatch continues only for `run_stopped`.
10. Build order: where data edges leave the order open, a subscriber is built **before** the node
    that emits to it, so its handler exists at the first emission. Data edges always win; when the
    two disagree the validator warns (`event_order`) and an event emitted to an unbuilt subscriber
    fails the run (`EventDeliveryError`).
11. A topic with no subscriber costs one dictionary lookup and builds no summary.

## Mirroring

Every dispatched event is reported through `ctx.reporter` as a frame
`{"kind":"event","name":"<port>","source":"<uid>","type":"EpochEnding","summary":{...},
"seq":<int>,"t":<float seconds since run start>}` on stream `node:<uid>`. Node state changes
use the frame kind `node_state` on stream `graph:run` (see `runs.md`).
