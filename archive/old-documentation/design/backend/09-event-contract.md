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
event is written as an `id: <cursor>` line followed by its `data:`
line (`_frame` in `presentation/sse.py`). Without that line the header
is absent and this note describes a browser that does not exist.

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

**Which is why the id on the wire is a cursor, not a number.** See §1a.

### 1a. The epoch: what makes "a previous process" detectable

A bare `seq` cannot say which process issued it, and the only way to spot
a foreign one was to notice that its number was *ahead* of everything this
process had published. That test works right up until the moment this
process publishes past it — and a client reconnecting late after a restart
is exactly that case. Measured on the round-3 reproduction: with 60 events
published, `replay_since(40)` returned `complete=True` and events 41..60,
so a tab was told its history was continuous when it was not, and skipped
the refetch that would have corrected it.

So the bus mints an **epoch** per process — `uuid4().hex[:12]`, never
derived from the clock, because a machine that restarts twice in one
second would repeat a timestamp — and the id on the wire is
`"{epoch}:{seq}"`. A cursor whose epoch is not ours is refused outright,
whatever its number. That is the whole fix, and it costs the client
nothing: `EventSource` hands the header back verbatim and never parses it,
so the epoch is opaque to the browser and the frontend never sees it.

The bare `seq` is still in the JSON payload, for a reader that only
consumes `data`. Only the framed id is used for reconnection, and only the
bus reads it. A header that is absent, malformed, or a bare number — a
client predating this — is treated as no header at all, because a number
alone cannot be acted on.

One consequence worth naming: `replayed_through`, the watermark the live
path drops up to, falls back to the client's own cursor **only when the
epoch matches**. A foreign cursor says nothing about how far into *this*
stream the client has read, so defaulting to its number would make the
live path discard the new stream's opening events.

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
| a cursor for this epoch, `seq` still in the ring | replay everything after it, then go live |
| a cursor for this epoch, `seq` older than the ring | `resync_required`, then go live |
| a cursor for another epoch, at any `seq` | `resync_required`, then go live |
| something unparsable, including a bare number | treated as nothing |

The last three cases are not an error. The client cannot be given a complete
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

The bug class is silent and has no other guard: the backend renames a
field, every frame stops carrying it, and the frontend reads `undefined`
and renders an em dash or a blank — with no error anywhere. Line coverage
is green on both sides. Mutation testing on `sse.py` found the same class
of thing by accident, in the form of surviving string-literal mutations
for header names nobody asserts.

`presentation/event_schema.py` generates a JSON Schema per event from the
dataclasses; `tests/test_event_contract.py` does three things with it.

**The schema describes the wire, not the dataclass.** The first draft
generated from the dataclasses alone was wrong, and the test caught it:
the frame a client receives also carries `type`, `seq`, and — for a
diverged loss — `nonfinite`, none of which are declared on any event.
`frontend/js/views/run.js` reads `e.nonfinite` specifically so a NaN loss
renders as an error rather than as a blank, so a schema without it would
have flagged the one field most worth having. Those three keys belong to
this layer, so the generator lives here rather than in `domain/`.

**Real frames are checked against it.** Every event is constructed, run
through the actual serializer, and validated against its own schema — so a
field added to a dataclass, or a key added by the serializer, shows up as
a failure rather than as silence. The validator is hand-written because
`jsonschema` is not a dependency of this project (it is here only as a
transitive dependency of ComfyUI's `matrix-nio`). It raises on any
construct it does not understand rather than passing it, and the test
cross-checks it against the real `jsonschema` on **70** payload/schema
pairs whenever that package is importable — so the subset is verified,
not merely intended.

**The frontend is checked against the schema.** Which is the part worth
having. Two details decide whether it produces signal or noise:

- A variable counts as a frame only if it is compared `.type`-against a
  wire type name. Without that, `opt.type === "checkbox"` in
  `views/config.js` and `input.type = "number"` are the same shape as a
  frame read, and a naive scan drowns in them.
- Reads are checked against the **union** of every event's fields, not
  per event type. So a field renamed on one event while still present on
  another would pass. That gap is real and is not worth closing with
  brace-matching regex over JavaScript: the failure this must catch is a
  name that exists *nowhere*, which the union catches exactly.

Verified by renaming `RunProgressed.cache_total` and watching the check
report `frontend/js/views/run.js: ['e'].cache_total` — then reverting.