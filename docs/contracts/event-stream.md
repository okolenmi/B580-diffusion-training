# Contract: event stream

Python: `backend/presentation/sse.py`, `backend/infrastructure/events/callback_event_bus.py`, `backend/application/ports/event_bus.py`,
`backend/application/event_delivery.py`, `backend/domain/events.py`. Frontend: `frontend/js/api.js`, `frontend/js/lib/events.js`.

This is the **server-to-browser** stream, `GET /api/v1/events` (`text/event-stream`). The node-to-node event plane inside a run is a different
thing (`events.md`); the two meet only in that the supervisor publishes a graph execution's lifecycle here.

## Frames

Each frame is one JSON object in `data:`. Fields come from the event's dataclass (`backend/domain/events.py`) plus keys added on the wire:

| Key | Added by | Meaning |
|---|---|---|
| `type` | `sse.serialize_event` | the event's wire name (`graph_execution_started`, ...) |
| `seq` | `sse.serialize_event` | position in this process's stream, from 1 |
| `nonfinite` | `json_safe.sanitize` | only when a payload held a non-finite float: the value is `null` and `{key: "nan" \| "inf" \| "-inf"}` says what it was |

Events today: `graph_execution_queued`, `_started`, `_progressed`, `_finished`, `_failed`, `_stopped`, `graph_executions_deleted`.
`stream_opened` is the first frame of every connection (below). A line `: ping` every 15 s (`SSE_HEARTBEAT_SECONDS`) keeps proxies from closing an
idle stream and lets the server notice a dead client.

## Position: `seq`, epoch, cursor

- `seq` is assigned by the bus at publish time, is **per process** and is **not durable**. After a restart it starts again at 1.
- The bus mints an **epoch** per process (`uuid4().hex[:12]`, never the clock: two restarts in one second must not repeat it).
- The SSE `id:` of a tracked frame is the **cursor** `"{epoch}:{seq}"`. `EventSource` returns it verbatim as `Last-Event-ID` on automatic
  reconnect and never parses it, so the epoch is opaque to the frontend. Without an `id:` line the browser would send nothing.
- Why the epoch: a bare number cannot say which process issued it. After a restart, a client's `Last-Event-ID: 40` looks valid as soon as the new
  process has published past 40, so it was told its history was continuous when it was not.
- A header that is absent, malformed or a bare number counts as no header.

## Reconnect

The server **subscribes first, then replays**, so nothing published in between is lost; the overlap is delivered twice and the live path drops
anything with `seq` at or below what the replay covered (`replayed_through`).

| Client sends | Answer |
|---|---|
| nothing (first connect), or an unusable value | `stream_opened` with `resync_required: true`, then live |
| a cursor of this epoch, `seq` still in the ring | `stream_opened` with `resync_required: false`, the missed lifecycle events, then live |
| a cursor of this epoch, `seq` older than the ring | `resync_required: true`, then live |
| a cursor of another epoch, any `seq` | `resync_required: true`, then live |

`stream_opened` carries `{type, occurred_at, resync_required, replayed_through}`. `resync_required` means "refetch the source of truth" (the
database rows); it is about **lifecycle** events only, because a missed delta is not recoverable by refetching. `replayed_through` is the
end of the replay (or the client's own `seq` when its epoch matched), not the id it sent.

The ring (`EVENT_REPLAY_RING`, 512) holds **lifecycle events only**. Nothing is persisted: a client away longer than the ring resyncs instead of
replaying. A replayed event the client already saw is possible after a stale `Last-Event-ID`; handlers patch rows from the payload, so it is
harmless by construction.

## Delivery classes

`delivery_class(event_type)` (`event_delivery.py`) is the single answer, used by the bus (what to ring-buffer) and by the per-client buffer
(what may be dropped).

| Class | Members | Policy |
|---|---|---|
| lifecycle | everything not listed below | never coalesced, never dropped for a newer event of another kind |
| delta | `graph_execution_progressed` | only meaningful in order: never coalesced, never evicted |
| state | none today | a newer sample for the same key supersedes a queued one |

Unlisted kinds are lifecycle on purpose: wrongly treating an event as coalescible loses a terminal event, the other mistake costs one frame.

Each client has a bounded buffer (`SSE_QUEUE_MAX`, 256). Only when it is full and nothing can be coalesced is the oldest frame dropped, whatever
its class; every drop is counted and logged. The frontend refetches on every (re)open and the dashboard polls slowly, which closes that gap.

## The schema test

`backend/presentation/event_schema.py` generates a JSON Schema per event **from the dataclasses plus the wire keys above** (the dataclasses alone
describe a frame that is never received). `backend/tests/test_event_contract.py`:

1. builds every event, runs it through the real serializer and validates it against its schema, so a new field or a new serializer key fails the
   test instead of passing silently;
2. checks the frontend: a variable counts as a frame only when it is compared by `.type` against a wire type name (otherwise `input.type ===
   "number"` would count), and the fields it reads must exist in the **union** of all events' fields. A name that exists nowhere fails; a field
   renamed on one event but still present on another does not (closing that gap needs a JavaScript parser, not a regex);
3. cross-checks the hand-written validator against `jsonschema` when that package is importable (`jsonschema` is not a project dependency).

The failure this guards against has no other guard: the backend renames a field, the frontend reads `undefined` and draws a blank, and line
coverage is green on both sides.

## Changing it

Contract page first, then the dataclass, the schema test and the frontend in the same commit (`../developing.md`, "Changing a public contract").
