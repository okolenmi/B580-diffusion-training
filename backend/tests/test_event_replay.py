"""Unit tests -- event sequence numbers, the replay ring, and Last-Event-ID.

The bug this exists for: `/api/v1/events` had no replay, so a browser tab
that slept through `graph_execution_finished` came back still believing it was
running. The frontend papered over it by refetching on every reconnect,
which was racy and cost a full fetch per reconnect
(`docs/design/backend/09-event-contract.md`).

Run directly: python backend/tests/test_event_replay.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.event_delivery import is_lifecycle
from backend.application.limits import EVENT_REPLAY_RING
from backend.domain.events import (
    GraphExecutionFinished,
    GraphExecutionProgressed,
    GraphExecutionQueued,
    GraphExecutionStarted,
)
from backend.application.ports.event_bus import EventCursor
from backend.infrastructure.events.callback_event_bus import CallbackEventBus
from backend.tests.support import check, finish

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

#: Generous next to the ~1ms these take, and far below the 15s SSE
#: heartbeat -- so a hang here is this test's bug, not the stream's.
STREAM_TIMEOUT_SECONDS = 5.0


def queued(execution_id: int = 1) -> GraphExecutionQueued:
    return GraphExecutionQueued(execution_id=execution_id, node_count=4,
                                occurred_at=NOW)


def started(execution_id: int = 1) -> GraphExecutionStarted:
    return GraphExecutionStarted(execution_id=execution_id, occurred_at=NOW)


def completed(execution_id: int = 1, nodes: int = 4) -> GraphExecutionFinished:
    return GraphExecutionFinished(execution_id=execution_id, nodes=nodes,
                                  occurred_at=NOW)


def node_progressed(node_id: str = "nA") -> GraphExecutionProgressed:
    return GraphExecutionProgressed(
        execution_id=1, node_id=node_id, ok=True, duration_ms=12.5, occurred_at=NOW,
    )


# --------------------------------------------------------------------------
# the bus
# --------------------------------------------------------------------------

def test_sequence_numbers() -> None:
    print("\n== bus: every event gets a number, increasing, from 1 ==")
    bus = CallbackEventBus()
    seen = []
    bus.subscribe(seen.append)

    bus.publish(started())
    bus.publish(node_progressed())
    bus.publish(completed())
    bus.publish(node_progressed())

    check([item.seq for item in seen] == [1, 2, 3, 4],
          f"seqs are 1..4 in publish order (got {[i.seq for i in seen]})")
    check([item.event_type for item in seen] == [
        "graph_execution_started", "graph_execution_progressed",
        "graph_execution_finished",
        "graph_execution_progressed",
    ], "the order the publisher used is preserved")
    check(bus.last_seq == 4, f"last_seq tracks it (got {bus.last_seq})")
    check(all(item.event is not None for item in seen), "each carries its event")
    check(seen[2].event_type == "graph_execution_finished",
          "event_type is readable without unwrapping first")


def _cursor(bus, seq: int) -> EventCursor:
    """The cursor a client of *this* bus would hold after seeing ``seq``."""
    return EventCursor(bus.epoch, seq)


def _dead_cursor(seq: int) -> EventCursor:
    """A cursor of the same shape, issued by a process that no longer runs.

    Not a random string: a fixed one so a reader can see that nothing
    depends on its value, only on its being different.
    """
    return EventCursor("0000deadbeef", seq)


def test_ring_holds_lifecycle_only() -> None:
    print("\n== bus: the ring keeps lifecycle events and nothing else ==")
    bus = CallbackEventBus()
    bus.publish(started())
    bus.publish(node_progressed())
    bus.publish(completed())
    bus.publish(node_progressed())

    replayed = [item.event_type for item in bus.replay_since(_cursor(bus, 0)).events]
    check(replayed == ["graph_execution_started", "graph_execution_finished"],
          f"a state sample and a delta are not ringed (got {replayed})")
    check(is_lifecycle("graph_execution_finished")
          and not is_lifecycle("graph_execution_progressed"),
          "and the exclusion is the delivery class, not a second list")
    check(
        all(item.seq <= 4 for item in bus.replay_since(_cursor(bus, 0)).events),
        "ringed items keep the numbers they were published with",
    )


def test_replay_since_truth_table() -> None:
    print("\n== bus: the four answers replay_since can give ==")
    bus = CallbackEventBus()
    bus.publish(started())      # 1
    bus.publish(node_progressed())   # 2
    bus.publish(completed())    # 3

    up_to_date = bus.replay_since(_cursor(bus, 3))
    check(up_to_date.events == () and up_to_date.complete,
          "a client that saw everything gets nothing, and is told so")

    behind = bus.replay_since(_cursor(bus, 1))
    check([item.seq for item in behind.events] == [3],
          f"a client behind gets exactly what it missed (got {behind.events})")
    check(behind.complete, "and that replay is complete")

    other_process = bus.replay_since(_dead_cursor(2))
    check(other_process.events == () and not other_process.complete,
          f"a cursor from a previous process gets NO plausible-looking "
          f"replay, even at seq 2 which this process has also reached "
          f"(got complete={other_process.complete}, "
          f"{[i.seq for i in other_process.events]})")

    never = bus.replay_since(_cursor(bus, 0))
    check([item.seq for item in never.events] == [1, 3],
          "seq 0 asks for everything still ringed")
    check(never.complete, "seq 0 is adjacent to the oldest ring entry")

    nothing = bus.replay_since(None)
    check(nothing.events == () and not nothing.complete,
          "and a client that sent nothing usable gets the same honest no")


def test_ring_eviction_is_honest() -> None:
    print("\n== bus: past the ring, the answer is resync, not a short answer ==")
    bus = CallbackEventBus(replay_size=4)
    for index in range(10):
        bus.publish(completed(execution_id=index))

    check(len(bus.replay_since(_cursor(bus, 0)).events) == 4, "the ring is bounded")
    stale = bus.replay_since(_cursor(bus, 1))          # 2..6 were evicted
    check(not stale.complete,
          "a client id older than the ring is told the replay has a hole")
    check(
        [item.seq for item in stale.events] == [7, 8, 9, 10],
        "it still gets what IS there -- silently returning nothing would "
        "read as 'nothing happened'",
    )
    recent = bus.replay_since(_cursor(bus, 7))
    check(recent.complete and [i.seq for i in recent.events] == [8, 9, 10],
          "a client inside the window gets a complete answer")

    check(EVENT_REPLAY_RING >= 64,
          f"the shipped ring is not token ({EVENT_REPLAY_RING})")


def test_empty_bus_never_claims_completeness() -> None:
    print("\n== bus: no history means no promise ==")
    bus = CallbackEventBus()
    empty = bus.replay_since(_cursor(bus, 1))
    check(empty.events == () and not empty.complete,
          "a bus that never ringed anything cannot promise a complete replay")

    bus.publish(started())
    bus.publish(completed())
    check(bus.replay_since(_cursor(bus, 1)).complete,
          "but once something is ringed, the client's position is known")


# --------------------------------------------------------------------------
# the stream
# --------------------------------------------------------------------------

class _RacingBus(CallbackEventBus):
    """A bus that publishes *during* ``replay_since``.

    That is the exact window `event_stream` has to survive: the
    subscription is already attached, so the event is destined for the
    live buffer, and the ring read that follows will also see it. Both
    paths can therefore deliver the same event, which is why the stream
    drops anything the replay already covered. Doing it from a thread
    would be a coin flip; doing it here makes the interleaving certain.
    """

    def __init__(self, event) -> None:
        super().__init__()
        self._racing = event
        self.raced = False

    def replay_since(self, cursor):
        if not self.raced:
            self.raced = True
            self.publish(self._racing)
        return super().replay_since(cursor)


async def _stream(app, *, headers=(), stop_after) -> bytes:
    """Drive one GET /api/v1/events to completion, return the raw body.

    `stop_after(body) -> bool` ends the stream by making the next receive
    report a disconnect. The app returns when the generator notices, so
    a hang in here surfaces as a TimeoutError instead of hanging the run.
    """
    chunks: list[bytes] = []
    finished = asyncio.Event()

    async def receive():
        if finished.is_set():
            return {"type": "http.disconnect"}
        await asyncio.sleep(3600)  # cancellable, peeked inside the generator
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            if stop_after(b"".join(chunks)):
                finished.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/v1/events",
        "raw_path": b"/api/v1/events",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"localhost"),
            (b"accept", b"text/event-stream"),
            *headers,
        ],
        "client": ("1.2.3.4", 1234),
        "server": ("localhost", 8766),
    }
    # asyncio.timeout rather than a `timeout` parameter: a hung stream
    # should fail here as a TimeoutError, not as a parameter some caller
    # forgot to raise.
    async with asyncio.timeout(STREAM_TIMEOUT_SECONDS):
        await app(scope, receive, send)
    return b"".join(chunks)


def _frames(body: bytes) -> list[dict]:
    """Every `data:` payload in an SSE body, parsed."""
    out = []
    for line in body.decode().splitlines():
        if line.startswith("data: "):
            out.append(json.loads(line[6:]))
    return out


def _ids(body: bytes) -> list[int]:
    """The sequence number of every SSE `id:`, in order.

    The wire form is ``"{epoch}:{seq}"`` -- what EventSource would track
    and hand back verbatim -- and these are the numbers inside it. Which
    epoch it was is checked separately by `_epochs`, because "the frames
    are numbered 3, 4" and "the frames came from the process this client
    is talking to" are different claims.
    """
    out = []
    for line in body.decode().splitlines():
        if line.startswith("id: "):
            out.append(int(line[4:].partition(":")[2]))
    return out


def _wire_ids(body: bytes) -> list[str]:
    """Every framed id verbatim, so a test can compare the whole cursor."""
    return [
        line[4:]
        for line in body.decode().splitlines()
        if line.startswith("id: ")
    ]


def _build_stream_app(bus):
    """The real /events endpoint over `bus`, with no other machinery.

    `event_stream` is the unit under test; going through the full app
    would only add unrelated routes to drive. The heartbeat is irrelevant
    here because every case stops on a frame it expects.
    """
    from starlette.applications import Starlette
    from starlette.routing import Route

    from backend.presentation.sse import event_stream

    class _Request:
        def __init__(self, real):
            self._real = real

        def __getattr__(self, name):
            return getattr(self._real, name)

    async def events(request):
        return await event_stream(bus, _Request(request))

    app = Starlette(routes=[Route("/api/v1/events", events)])
    return app


def _saw(*types: str):
    """End the stream once every named event type has been written."""
    return lambda body: all(f'"type": "{kind}"'.encode() in body for kind in types)


def _n_frames(count: int):
    """End the stream once ``count`` ``data:`` lines have been written.

    Used where the *number* of replayed frames is what varies. Counting
    beats waiting for a named type: a case that correctly replays nothing
    would otherwise sit until the 15s heartbeat and then time out, which
    is a test about the timeout rather than about replay.

    Safe to stop on a count because nothing else publishes during these
    scenarios -- the generator writes the opening frame and the whole
    replay before it awaits anything, so `count` is the exact total.
    """
    def predicate(body: bytes) -> bool:
        text = body.decode()
        return text.count("data: ") >= count
    return predicate


def test_first_connect_asks_for_a_resync() -> None:
    print("\n== stream: a first connect has missed everything, and says so ==")
    bus = CallbackEventBus()
    bus.publish(started())
    bus.publish(completed())
    body = asyncio.run(_stream(
        _build_stream_app(bus),
        headers=(),
        stop_after=_saw("stream_opened"),
    ))
    frames = _frames(body)
    check(frames[0]["type"] == "stream_opened", "the opening frame comes first")
    check(frames[0]["resync_required"] is True,
          f"a first connect cannot have replayed anything (got "
          f"{frames[0]['resync_required']!r})")
    check(frames[0]["replayed_through"] is None,
          "and it has no position to report -- it was given nothing")
    check(
        not any(item["type"] in ("graph_execution_finished",
                                 "graph_execution_started")
                for item in frames),
        "no history is replayed to a client that did not ask for it",
    )
    check(_ids(body) == [],
          "stream_opened carries no id: line -- it is not a bus event, and an "
          "id here would move the client's Last-Event-ID backwards")


def test_replay_on_reconnect() -> None:
    print("\n== stream: Last-Event-ID asks for the gap, and gets it ==")
    bus = CallbackEventBus()
    bus.publish(started())      # 1
    bus.publish(node_progressed())   # 2
    bus.publish(completed())    # 3
    bus.publish(started(2))     # 4

    body = asyncio.run(_stream(
        _build_stream_app(bus),
        headers=[(b"last-event-id", _cursor(bus, 2).wire().encode())],
        stop_after=_saw("graph_execution_started"),
    ))
    frames = _frames(body)
    check(frames[0]["type"] == "stream_opened", "the opening frame comes first")
    check(frames[0]["resync_required"] is False,
          f"the gap was covered, so no refetch is needed (got "
          f"{frames[0]['resync_required']!r})")
    check(all(w.startswith(f"{bus.epoch}:") for w in _wire_ids(body)),
          f"every framed id carries this process's epoch, so the browser "
          f"hands back something this server can check (got "
          f"{_wire_ids(body)})")
    check(frames[0]["replayed_through"] == 4,
          f"the frame says where the client ends up, not where it started "
          f"(got {frames[0]['replayed_through']!r} -- it sent 2, and the "
          f"replay below carries 3 and 4)")
    replayed = [f["type"] for f in frames[1:]]
    check(replayed == ["graph_execution_finished", "graph_execution_started"],
          f"exactly the two missed events, in order (got {replayed})")
    check(_ids(body) == [3, 4],
          f"each replayed frame carries its SSE id: line (got {_ids(body)})")
    check(
        all(f.get("seq") == i for f, i in zip(frames[1:], [3, 4], strict=True)),
        "and the same seq inside the payload, for readers that skip framing",
    )
    check(
        not any(f["type"] == "graph_execution_progressed" for f in frames),
        "the delta at seq 2 is not replayed -- the client already had it, "
        "and a superseded node sample is worse than none",
    )


def test_reconnect_cannot_be_covered() -> None:
    print("\n== stream: an uncoverable gap says so instead of guessing ==")
    bus = CallbackEventBus(replay_size=2)
    bus.publish(started())
    bus.publish(node_progressed())
    bus.publish(completed())     # 3
    bus.publish(started(2))      # 4

    def connect(last_event_id: bytes, expected: int) -> tuple[list[dict], bytes]:
        body = asyncio.run(_stream(
            _build_stream_app(bus),
            headers=[(b"last-event-id", last_event_id)],
            # expected counts the opening frame plus whatever is replayed.
            stop_after=_n_frames(expected),
        ))
        return _frames(body), body

    # A cursor from a process that no longer runs. Replaying anything would
    # be a lie: seq 4 here is a different event from seq 4 over there.
    #
    # The number is deliberately one this process *has* reached. That is
    # the case that used to be missed: before the cursor carried an epoch,
    # a foreign id was only recognisable while it was ahead of everything
    # published here, so once this process passed it the id looked
    # current. Asking for 2 on a bus holding 4 events was served as a
    # normal "give me 3 and 4" -- complete, and wrong.
    previous, previous_body = connect(_dead_cursor(2).wire().encode(),
                                      expected=1)  # opening only
    check(previous[0]["resync_required"] is True,
          f"a cursor from a previous process -> resync_required even when "
          f"its number is inside this process's range (got "
          f"{previous[0]['resync_required']!r})")
    check(_ids(previous_body) == [],
          f"and nothing is replayed: its seqs describe a different stream "
          f"(got ids {_ids(previous_body)})")
    check(previous[0]["replayed_through"] is None,
          "and no high-water mark is claimed, because nothing was replayed")

    # An id older than the ring: the hole is real, so it is flagged -- and
    # the tail the ring *does* hold is still sent, because the flag rides
    # in the frame above it and a client that honours it refetches anyway.
    # Sending nothing would throw away information for no gain.
    stale, stale_body = connect(_cursor(bus, 1).wire().encode(),
                              expected=3)     # opening + 2 replayed
    check(stale[0]["resync_required"] is True,
          f"an id older than the ring -> resync_required (got "
          f"{stale[0]['resync_required']!r})")
    check(_ids(stale_body) == [3, 4],
          f"the tail the ring still holds is sent, and it is honest about "
          f"the hole via the flag above it (got ids {_ids(stale_body)})")
    check(stale[0]["replayed_through"] == 4,
          "the frame names the high-water mark the replay reached")
    check(
        stale[0]["resync_required"] is True and _ids(stale_body),
        "flagged incomplete AND still carrying what is there -- a client "
        "refetches, and is not also denied the events we do have",
    )


def test_unusable_last_event_id_is_ignored() -> None:
    print("\n== stream: a junk Last-Event-ID is treated as no header ==")
    bus = CallbackEventBus()
    bus.publish(started())
    bus.publish(completed())
    # A bare number is in this list on purpose. It was a valid id until the
    # cursor gained an epoch, and it is now indistinguishable from a client
    # guessing -- which is the truth: a number cannot say which process
    # issued it, so it is treated as no header at all rather than trusted.
    for junk in (b"", b"abc", b"-1", b"0", b"1.5", b"9e9", b"  ", b"\xff",
                 b"2", b"9999", f"{bus.epoch}".encode(),
                 f"{bus.epoch}:".encode(), f"{bus.epoch}:0".encode(),
                 f"{bus.epoch}:-1".encode(), f"{bus.epoch}:x".encode()):
        body = asyncio.run(_stream(
            _build_stream_app(bus),
            headers=[(b"last-event-id", junk)],
            stop_after=_saw("stream_opened"),
        ))
        frames = _frames(body)
        check(frames[0]["type"] == "stream_opened" and
              frames[0]["resync_required"] is True,
              f"Last-Event-ID {junk!r} -> treated as absent, no crash")


def test_subscribe_then_replay_race() -> None:
    print("\n== stream: an event published mid-replay arrives exactly once ==")
    bus = _RacingBus(completed(execution_id=7, nodes=3))
    bus.publish(started())  # 1 -- what the client already has

    body = asyncio.run(_stream(
        _build_stream_app(bus),
        headers=[(b"last-event-id", _cursor(bus, 1).wire().encode())],
        stop_after=_saw("graph_execution_finished"),
    ))
    frames = _frames(body)
    delivered = [f for f in frames if f["type"] == "graph_execution_finished"]
    check(len(delivered) == 1,
          f"the raced event is not delivered twice (got {len(delivered)})")
    ids = _ids(body)
    check(len(ids) == len(set(ids)),
          f"and no sequence number appears twice in one stream (got {ids})")
    check(frames[0]["resync_required"] is False,
          "a replay that covers the client is still reported as complete")


def main() -> None:
    test_sequence_numbers()
    test_ring_holds_lifecycle_only()
    test_replay_since_truth_table()
    test_ring_eviction_is_honest()
    test_empty_bus_never_claims_completeness()
    test_first_connect_asks_for_a_resync()
    test_replay_on_reconnect()
    test_reconnect_cannot_be_covered()
    test_unusable_last_event_id_is_ignored()
    test_subscribe_then_replay_race()
    finish()


if __name__ == "__main__":
    main()