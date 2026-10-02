"""Unit tests -- the JSON boundary (docs 07 F-03, F-09).

Pins the three places where a payload becomes bytes: the walker in
``backend/json_safe.py``, the SSE frame serializer and its client
buffer, and the monitor-bus adapter (which frames node telemetry).

Run directly: python backend/tests/test_json_safe.py
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.events import GraphExecutionProgressed
from backend.infrastructure.monitor_bus import SharedMonitorBus
from backend.json_safe import sanitize, strict_dumps
from backend.application import event_delivery
from backend.presentation.sse import (
    ClientBuffer,
    coalesce_key,
    delivery_class,
    serialize_event,
)
from backend.tests.support import check, finish

NAN = float("nan")
INF = float("inf")


def strict_loads(text: str) -> object:
    """Parse like a browser: NaN/Infinity are a syntax error."""

    def reject(constant: str):
        raise ValueError(f"invalid JSON constant {constant}")

    return json.loads(text, parse_constant=reject)


def test_walker() -> None:
    print("\n== sanitize: non-finite floats -> null + named keys ==")
    clean = sanitize(
        {
            "loss": NAN,
            "avg_loss": INF,
            "best": -INF,
            "step": 7,
            "name": "ok",
            "missing": None,
            "flag": True,
            "nested": {"values": {"peak_vram": NAN}, "ok": 1.5},
            "series": [1.0, NAN],
        }
    )
    check(clean["loss"] is None and clean["avg_loss"] is None, "floats become null")
    check(clean["best"] is None, "-inf becomes null too")
    check(
        clean["nonfinite"] == {"loss": "nan", "avg_loss": "inf", "best": "-inf"},
        f"every replaced key is named with its kind (got {clean['nonfinite']})",
    )
    check(
        clean["step"] == 7 and clean["name"] == "ok" and clean["flag"] is True,
        "ordinary values pass through untouched",
    )
    check(clean["missing"] is None, "a real null stays null")
    check(clean["series"] == [1.0, None], "a bare array is cleaned, nulled")
    check(
        clean["nested"]["values"]["nonfinite"] == {"peak_vram": "nan"},
        f"nested objects name their own key (got {clean['nested']['values']})",
    )
    check(
        clean["nested"]["ok"] == 1.5 and "nonfinite" not in clean["nested"],
        "a sibling without a bad value gets no marker",
    )

    # A list of objects: each item is self-describing (this is what
    # makes {"runs": [...]} renderable without path arithmetic).
    listed = sanitize({"runs": [{"id": 1, "current_loss": INF}, {"id": 2}]})
    check(
        listed["runs"][0]["nonfinite"] == {"current_loss": "inf"},
        f"list items name their own field (got {listed['runs'][0]})",
    )
    check(
        "nonfinite" not in listed["runs"][1] and "nonfinite" not in listed,
        "no marker where nothing was non-finite",
    )
    check("nonfinite" not in sanitize({"loss": 0.5}), "clean payload unchanged")


def test_strict_dumps_guard() -> None:
    print("\n== strict_dumps: the invariant guard ==")
    text = strict_dumps(sanitize({"loss": NAN, "step": 3}))
    check(strict_loads(text) == {"loss": None, "step": 3, "nonfinite": {"loss": "nan"}},
          f"sanitized output is strict JSON ({text})")
    try:
        strict_dumps({"loss": NAN})
    except ValueError:
        check(True, "an unsanitized non-finite float raises instead of shipping NaN")
    else:
        check(False, "strict_dumps must refuse a bare NaN")


def test_serialize_event_nonfinite() -> None:
    # The reproduction of docs 07 F-03, rebuilt on the events that still
    # exist. It needs a non-finite *float field*, which after the run
    # events' removal is `duration_ms`; nothing about the bug is specific
    # to what the field means, only to it being a float that diverged.
    print("\n== SSE serializer: a diverged value never ships bare NaN ==")
    event = GraphExecutionProgressed(
        execution_id=1, node_id="nA", ok=True, duration_ms=NAN,
        occurred_at=dt.datetime.now(dt.UTC),
    )
    text = serialize_event(event)
    payload = strict_loads(text)  # raises == F-03 reproduces
    check(isinstance(payload, dict), "frame parses with a strict parser")
    check(payload["duration_ms"] is None,
          f"the diverged float is null (got {payload['duration_ms']!r})")
    check(
        payload["nonfinite"] == {"duration_ms": "nan"},
        f"its path is named with its kind (got {payload.get('nonfinite')})",
    )
    check(
        payload["node_id"] == "nA" and payload["ok"] is True
        and payload["type"] == "graph_execution_progressed",
        "the finite fields of the same frame survive",
    )

    inf_event = GraphExecutionProgressed(
        execution_id=1, node_id="nB", ok=True, duration_ms=INF,
        occurred_at=dt.datetime.now(dt.UTC),
    )
    inf_payload = strict_loads(serialize_event(inf_event))
    check(inf_payload["nonfinite"] == {"duration_ms": "inf"},
          "infinity is distinguished from NaN in the marker")
    check("NaN" not in text and "Infinity" not in text, "no bare tokens in the bytes")


def test_client_buffer() -> None:
    """Coalescing is decided by delivery class (docs 07 F-09, corrected by
    docs 08 N-04). Three classes, three rules:

    state -- coalesce per key, so a newer sample for the same subject
      supersedes only that subject's queued sample. **No production event
      is currently a state event** -- `run_progressed` went with the
      supervised-subprocess route -- so this half is exercised with a
      declared-but-unused kind. That is deliberate: the rule is real code
      with real consequences (an overflow that coalesces the wrong thing
      loses frames silently), and testing it only when an event happens to
      fit is how it rots.
    delta -- never coalesced and never evicted, because every node's
      completion is a fact that happened and all of them must arrive;
    lifecycle -- never dropped for the sake of a newer frame of any
      other kind, and only sacrificed at a full buffer once state and
      delta have both been tried.
    """

    # A state kind that nothing currently emits. See the docstring: this is
    # the coalescing rule under test, not a claim that anything emits it.
    #
    # It has to be put *into the table* to be a state kind -- `delivery_class`
    # reads the module global, so declaring it here is what makes the
    # coalescing path reachable at all. Restored immediately after, and the
    # fact that the table is empty in production is asserted at the end, so
    # this cannot quietly become a second, hidden state event.
    STATE_KIND = "hypothetical_state_sample"
    original_states = set(event_delivery.STATE_EVENT_TYPES)
    event_delivery.STATE_EVENT_TYPES = frozenset(
        original_states | {STATE_KIND}
    )

    async def scenario() -> None:
        async def next_frame(buf: ClientBuffer) -> str | None:
            """``get`` yields ``(seq, payload)``; these checks are about
            payloads, and the sequence side is pinned separately below."""
            return (await buf.get(0.01))[1]

        buf = ClientBuffer(maxsize=16)
        check(await buf.get(0.01) == (None, None),
              "empty buffer times out as (None, None) -- the heartbeat path")

        # --- state: coalesce, keyed ---
        # Real payloads, because a state frame's coalescing key is
        # derived from its own run_id. That also means a hand-passed key
        # cannot rescue an unkeyable frame -- see the explicit-key case
        # below, which asserts exactly that.
        p1 = f'{{"type":"{STATE_KIND}","run_id":1,"step":1}}'
        p2 = f'{{"type":"{STATE_KIND}","run_id":1,"step":2}}'
        buf.put(STATE_KIND, p1, seq=1)
        buf.put(STATE_KIND, p2, seq=2)
        check(len(buf) == 1, f"same-subject samples coalesce to the newest (got {len(buf)})")
        check(buf.coalesced == 1, f"coalesced counter (got {buf.coalesced})")
        seq, payload = await buf.get(0.01)
        check(payload == p2, "the surviving frame is the newest")
        check(seq == 2,
              f"and carries the newest seq, not the superseded one's "
              f"(got {seq}) -- Last-Event-ID replay drops frames by seq, so "
              f"a stale seq would let a live frame be replayed as new")

        # A different run's newest sample is not superseded by this one's.
        other = f'{{"type":"{STATE_KIND}","run_id":2,"step":1}}'
        buf.put(STATE_KIND, other)
        buf.put(STATE_KIND, p1)
        check(len(buf) == 2, f"two subjects' samples coexist (got {len(buf)})")
        check([await next_frame(buf), await next_frame(buf)] == [other, p1],
              "and both subjects' samples arrive, in order")

        # --- delta: N-04. Six node events must arrive as six ---
        p3 = f'{{"type":"{STATE_KIND}","run_id":1,"step":3}}'
        buf.put(STATE_KIND, p3)
        buf.put("graph_execution_finished", "c1")
        for node in "ABCDEF":
            buf.put("graph_execution_progressed", f"n{node}")
        check(len(buf) == 8, f"1 state + 1 lifecycle + 6 deltas all queued (got {len(buf)})")
        got = [await next_frame(buf) for _ in range(8)]
        check(
            got == [p3, "c1", "nA", "nB", "nC", "nD", "nE", "nF"],
            f"every node event survives a state frame (got {got})",
        )
        check(buf.coalesced == 1, f"no coalescing happened among the deltas (got {buf.coalesced})")

        # A delta is never given a coalescing key, even if a caller
        # offers one -- "has a key" must not mean "may evict" (N-04).
        buf2 = ClientBuffer(maxsize=8)
        buf2.put("graph_execution_progressed", "nX", key="node:X")
        buf2.put("graph_execution_progressed", "nY", key="node:X")
        check(len(buf2) == 2, "an explicit key does not license coalescing a delta")
        check(buf2.coalesced == 0, "and the counter stays honest")

        # --- overflow order: state, then delta, then anything ---
        small = ClientBuffer(maxsize=2)
        small.put("graph_execution_progressed", "nA")
        small.put("run_completed", "c1")
        small.put("graph_execution_progressed", "nB")   # full: drop a delta
        check(small.dropped_delta == 1, f"a delta was sacrificed, counted (got {small.dropped_delta})")
        check(small.dropped == 0, "no lifecycle frame touched")
        small.put("run_completed", "c2")               # full: no state left, drop delta
        check(small.dropped_delta == 2, f"second delta sacrificed (got {small.dropped_delta})")

        only_lifecycle = ClientBuffer(maxsize=2)
        for kind in ("run_created", "run_started", "run_completed"):
            only_lifecycle.put(kind, kind)
        check(only_lifecycle.dropped == 1,
              f"all-lifecycle buffer: oldest evicted and counted (got {only_lifecycle.dropped})")
        check(only_lifecycle.dropped_delta == 0, "not miscounted as a delta loss")
        got = [await next_frame(only_lifecycle), await next_frame(only_lifecycle)]
        check(
            got == ["run_started", "run_completed"],
            f"newest lifecycle frames survive in order (got {got})",
        )

    asyncio.run(scenario())

    event_delivery.STATE_EVENT_TYPES = frozenset(original_states)
    check(not event_delivery.STATE_EVENT_TYPES,
          "no event is a state event in production -- the table this test "
          "temporarily extended is empty again, and stays empty until a "
          "periodic-sample event actually exists")


def test_delivery_class_assignment() -> None:
    """The class table is the policy; a wrong entry silently reintroduces
    the class of bug N-04 was, so it is pinned directly."""
    print("\n== SSE: delivery classes are what the buffer assumes (N-04) ==")
    check(delivery_class("graph_execution_progressed") == "delta",
          "per-node graph progress is a delta: every one must arrive")
    check(not event_delivery.STATE_EVENT_TYPES,
          "nothing is a state event any more -- run_progressed went with "
          "the subprocess route (got "
          f"{sorted(event_delivery.STATE_EVENT_TYPES)})")
    for kind in ("graph_execution_queued", "graph_execution_started",
                 "graph_execution_finished", "graph_execution_failed",
                 "graph_execution_stopped", "graph_executions_deleted",
                 "stream_opened"):
        check(delivery_class(kind) == "lifecycle",
              f"{kind} is lifecycle (got {delivery_class(kind)})")

    check(coalesce_key("graph_execution_progressed", '{"node_id":"a"}') is None,
          "a delta never gets a coalescing key")
    check(coalesce_key("graph_execution_finished", '{"execution_id":1}') is None,
          "a lifecycle event never gets one either")

    # The key-building half, with the state class forced on: `coalesce_key`
    # returns None for a non-state kind before it ever looks at the payload,
    # so this is the only way to reach -- and therefore the only way to
    # test -- the key rules themselves.
    original = event_delivery.STATE_EVENT_TYPES
    event_delivery.STATE_EVENT_TYPES = frozenset({"state_probe"})
    try:
        check(coalesce_key("state_probe", '{"run_id":7,"step":1}') == "run:7",
              "a state sample is keyed by its run")
        check(coalesce_key("state_probe", '{"step":1}') is None,
              "a state sample with no run_id is not coalesced rather than "
              "coalesced wrongly")
        check(coalesce_key("state_probe", "not json") is None,
              "an unparseable payload is not coalesced")
    finally:
        event_delivery.STATE_EVENT_TYPES = original


def test_monitor_frames_are_strict() -> None:
    print("\n== monitor frames: node telemetry is sanitized at the adapter ==")
    bus = SharedMonitorBus()
    bus.report(
        "m1",
        {"type": "step", "step": 7, "loss": NAN, "vram_reserved_mb": 4096.0},
    )

    async def replay(bus: SharedMonitorBus, monitor_id: str) -> str:
        queue = bus.subscribe(monitor_id)
        return queue.get_nowait()  # history replays into a fresh subscriber

    frame = asyncio.run(replay(bus, "m1"))
    text = frame[len("data: "):].strip()
    payload = strict_loads(text)
    check(payload["loss"] is None, f"NaN loss became null (got {text})")
    check(payload["nonfinite"] == {"loss": "nan"}, f"marker (got {payload.get('nonfinite')})")
    check(payload["vram_reserved_mb"] == 4096.0, "finite fields untouched")
    check(payload["step"] == 7, "step untouched")

    clean_bus = SharedMonitorBus()
    clean_bus.report("m2", {"type": "step", "step": 1, "loss": 0.5})
    clean_frame = asyncio.run(replay(clean_bus, "m2"))
    clean_payload = strict_loads(clean_frame[len("data: "):].strip())
    check(
        "nonfinite" not in clean_payload and clean_payload["loss"] == 0.5,
        f"a healthy report is byte-identical to before (got {clean_payload})",
    )


def main() -> None:
    test_walker()
    test_strict_dumps_guard()
    test_serialize_event_nonfinite()
    test_client_buffer()
    test_delivery_class_assignment()
    test_monitor_frames_are_strict()
    finish()


if __name__ == "__main__":
    main()