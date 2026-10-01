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

from backend.domain.events import RunProgressed
from backend.infrastructure.monitor_bus import SharedMonitorBus
from backend.json_safe import sanitize, strict_dumps
from backend.presentation.sse import ClientBuffer, serialize_event
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
    # r8's exact event -- this is the reproduction of docs 07 F-03.
    print("\n== SSE serializer: a diverged loss never ships bare NaN ==")
    event = RunProgressed(
        run_id=1, step=7, total_steps=100, loss=NAN, avg_loss=INF, lr=1e-4,
        phase="training", cache_done=None, cache_total=None,
        occurred_at=dt.datetime.now(dt.timezone.utc),
    )
    text = serialize_event(event)
    payload = strict_loads(text)  # raises == F-03 reproduces
    check(isinstance(payload, dict), "frame parses with a strict parser")
    check(payload["loss"] is None, f"loss is null (got {payload['loss']!r})")
    check(payload["avg_loss"] is None, "avg_loss is null")
    check(
        payload["nonfinite"] == {"loss": "nan", "avg_loss": "inf"},
        f"paths named with kinds (got {payload.get('nonfinite')})",
    )
    check(
        payload["step"] == 7 and payload["lr"] == 1e-4 and payload["type"] == "run_progressed",
        "the finite fields of the same frame survive",
    )
    check("NaN" not in text and "Infinity" not in text, "no bare tokens in the bytes")


def test_client_buffer() -> None:
    print("\n== SSE buffer: progress coalesces, lifecycle is kept (F-09) ==")

    async def scenario() -> None:
        buf = ClientBuffer(maxsize=3)
        check(await buf.get(0.01) is None, "empty buffer times out (heartbeat)")

        buf.put("run_progressed", "p1")
        buf.put("run_progressed", "p2")
        check(len(buf) == 1, f"progress coalesces to the newest (got {len(buf)})")
        check(buf.coalesced == 1, f"coalesced counter (got {buf.coalesced})")
        check(await buf.get(0.01) == "p2", "the surviving frame is the newest")

        # Lifecycle interleaved with progress keeps both, in order.
        buf.put("run_progressed", "p3")
        buf.put("run_completed", "c1")
        buf.put("run_progressed", "p4")
        got = [await buf.get(0.01), await buf.get(0.01)]
        check(got == ["c1", "p4"], f"lifecycle kept, progress replaced (got {got})")

        # A lifecycle frame arriving at a full buffer evicts progress
        # first, and only ever counts it as coalesced.
        buf.put("run_failed", "c2")
        buf.put("run_cancelled", "c3")
        buf.put("run_progressed", "p5")
        check(len(buf) == 3, f"buffer is full (got {len(buf)})")
        buf.put("run_completed", "c4")
        check(buf.dropped == 0, "no lifecycle frame dropped for progress")
        check(buf.coalesced == 2, f"progress eviction counted (got {buf.coalesced})")
        got = [await buf.get(0.01) for _ in range(3)]
        check(
            got == ["c2", "c3", "c4"],
            f"full buffer gave up its progress frame (got {got})",
        )

        # A backlog of pure lifecycle frames: bounded, oldest evicted,
        # counted and logged -- never silent, never unbounded.
        full = ClientBuffer(maxsize=2)
        for kind in ("run_created", "run_started", "run_completed"):
            full.put(kind, kind)
        check(full.dropped == 1, f"oldest lifecycle evicted, counted (got {full.dropped})")
        check(len(full) == 2, "buffer stays bounded")
        got = [await full.get(0.01), await full.get(0.01)]
        check(
            got == ["run_started", "run_completed"],
            f"newest lifecycle frames survive in order (got {got})",
        )

    asyncio.run(scenario())


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
    test_monitor_frames_are_strict()
    finish()


if __name__ == "__main__":
    main()