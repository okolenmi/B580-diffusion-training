# 09. Event contract: sequence numbers and replay

*[← backend design index](README.md)*

**Status: in progress.** Implemented in two steps, each behind the tests
that already exist. This note is the reasoning; the code is the reference.

## The gap

`/api/v1/events` has no replay. A client that is not connected when an
event is published never receives it, and there is no way to ask for what
it missed.

That is survivable for a *delta* — the next one supersedes it — and it is
not survivable for a *lifecycle* event. A browser tab that sleeps through
`run_completed` comes back believing the run is still running, and keeps
believing it until something else happens to refresh the row. This is why
the frontend resyncs on every (re)connect
(`frontend/js/lib/events.js`): the resync is a **workaround for the
missing replay**, not a design choice. It costs a full refetch per
reconnect, and it is still racy — an event published between the refetch
and the subscription is lost.

`EventSource` sends a `Last-Event-ID` header on automatic
reconnection — **but only if the server emitted an SSE `id:` field** for
the frames it wants tracked. This server did not, so nothing was ever
sent. The feature therefore requires the server to opt in: every bus
event is written as an `id: <seq>` line followed by its `data:` line
(`_frame` in `presentation/sse.py`). Without that line the header is
absent and this note describes a browser that does not exist.

The sequence is *also* inside the JSON payload. Redundant on purpose: it
costs ~12 bytes and it means a reader that only consumes `data` — the
tests, and any hand-written client — still has the id it needs.

## Decision

### 1. A monotonic `seq` per process

Every event carries `seq: int`, assigned at publish time by the bus, from
1. It is **process-local** and **not durable**:

* The server is single-process. Runs are subprocesses but they do not
  publish events; the supervisor in the server process does.
* Making it durable would mean reading a high-water mark from the
  database on every startup and coordinating with in-flight publishers —
  cost paid on a path that already has a cheaper correct answer.

**Consequence, stated because it is a real one:** after a restart `seq`
starts again at 1. A client reconnecting with `Last-Event-ID: 500` is
therefore asking about events from a *previous* process. It must not be
answered with a partial replay from the new one. It gets
`resync_required` and refetches, which is exactly what it would have done
without this feature. The alternative — pretending the numbers are
comparable — would hand a client a plausible-looking wrong replay.

### 2. A bounded replay ring for lifecycle events only

The last `REPLAY_RING` **lifecycle** events, in a deque, on the bus.
Deltas are not ring-buffered: they are coalesced per client anyway, and
the value of an old progress sample is negative.

Lifecycle events are the ones worth replaying precisely because they are
*not* coalescable — WP-01 made that the rule, so that a missed
`run_completed` cannot be dropped for a slow client. This extends the same
reasoning across a reconnect instead of within one connection.

### 3. `Last-Event-ID` is honoured, and its three cases

| Client sends | Answer |
|---|---|
| nothing (first connect) | stream live, as today |
| `seq` still in the ring | replay everything after it, then go live |
| `seq` older than the ring, or from a previous process | `resync_required`, then go live |

The third case is not an error. The client cannot be given a complete
replay, and telling it so is the only honest answer — the alternative is
a stream that starts mid-history and looks complete.

### 4. Delivery classes stay where they are

`delivery_class()` in `presentation/sse.py` is already the single answer
for state / delta / lifecycle, and WP-01 pinned its table with a test.
Replay adds no third notion; it uses the same one.

## What this does not do

* **It is not a durable log.** Nothing is persisted, so a client that was
  away for longer than the ring resyncs rather than replaying. For a
  single-user loopback tool whose DB *is* the source of truth, refetching
  one row beats replaying a journal nobody wrote.
* **It does not deduplicate.** A client that receives a replayed event it
  had already seen is possible if it reconnects with a stale
  `Last-Event-ID`. Handlers are already idempotent about state (they
  patch a row from the payload), so this is benign by construction rather
  than by protocol.
* **It does not replay deltas**, by design, per above.

## Cost

One `int` per event on the wire; one bounded deque on the bus. The
subscribe path gains a lookup and a possible burst of frames before it
goes live. The burst is bounded by the ring size, and it goes through the
*same* per-client `ClientBuffer` as live traffic, so a client that
cannot absorb the replay drops frames by its existing rules instead of
growing without bound.

## The other half: proving the frontend only reads fields that exist

Separately implemented, because it is a different kind of check. A JSON
Schema is generated from the event dataclasses, and a test parses the
frontend's handlers for `e.<field>` / `event.<field>` references and
fails on any field the schema does not declare.

The bug class is silent and has no other guard: the backend renames a
field, every frame stops carrying it, and the frontend reads `undefined`
and renders an em dash or a blank — with no error anywhere. Line coverage
is green on both sides. Mutation testing on `sse.py` found the same class
of thing by accident, in the form of surviving string-literal mutations
for header names nobody asserts.