"""Unit tests -- the parent/child event file that carries a graph run out of
the server process.

The properties pinned here are the ones whose failure is silent: a record
that is half-written must not be read as a whole one, a lost run must not
become a silent run, and a non-finite float must not make a frame the
browser cannot parse.

Run directly: python backend/tests/test_graph_event_stream.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.graph_event_stream import (
    EventKind,
    ExecutionEventTail,
    ExecutionEventWriter,
)
from backend.tests.support import check, finish


def _path(tmp: str, name: str = "events.jsonl") -> Path:
    return Path(tmp) / "run_1" / name


def test_round_trip() -> None:
    print("\n== what the child writes, the parent reads back ==")
    with tempfile.TemporaryDirectory() as tmp:
        path = _path(tmp)
        writer = ExecutionEventWriter(path)
        writer.node({"node_id": "nA", "ok": True, "outputs": {"x": 1},
                     "error": None, "duration_ms": 12.5})
        writer.monitor("mon-1", {"step": 3, "loss": 0.5})
        writer.outcome(error=None, results_count=1)
        writer.close()

        tail = ExecutionEventTail(path)
        events = tail.poll()
        check([e.kind for e in events] == [EventKind.NODE, EventKind.MONITOR,
                                           EventKind.OUTCOME],
              f"all three kinds come back in order (got {[e.kind for e in events]})")
        check(events[0].payload["node_id"] == "nA", "node payload intact")
        check(events[1].payload["monitor_id"] == "mon-1", "monitor id intact")
        check(events[1].payload["data"] == {"step": 3, "loss": 0.5}, "monitor data intact")
        check(events[2].payload["results_count"] == 1, "outcome intact")
        check(tail.poll() == [], "a second poll with nothing new returns nothing")

        # A second writer on the same path *appends*, it does not replace.
        # That is the contract, not an accident: it is what lets a
        # restarted server re-read a still-running child's history (see
        # test_reset_replays_from_the_start), and it is why a run's event
        # path has to be unique per execution -- the supervisor gets that
        # from the row id, and rows are deleted rather than reused.
        writer2 = ExecutionEventWriter(path)
        writer2.node({"node_id": "nB", "ok": False, "outputs": {},
                      "error": "BoomError: x", "duration_ms": 1.0})
        appended = [e.payload.get("node_id")
                    for e in ExecutionEventTail(path).poll()
                    if e.kind is EventKind.NODE]
        check(appended == ["nA", "nB"],
              f"the second writer added to the file rather than replacing it "
              f"(got {appended})")
        arrived = [e.payload.get("node_id") for e in tail.poll() if e.kind is EventKind.NODE]
        check(arrived == ["nB"],
              f"which is exactly the one new record (got {arrived})")
        check(tail.poll() == [], "and nothing on a third poll")


def test_torn_tail_is_withheld() -> None:
    print("\n== a half-written line is not a record ==")
    with tempfile.TemporaryDirectory() as tmp:
        path = _path(tmp)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": "node", "node_id": "whole"}) + "\n")
            handle.write('{"kind": "node", "node_id": "half-writ')
        tail = ExecutionEventTail(path)
        first = tail.poll()
        check([e.payload["node_id"] for e in first] == ["whole"],
              f"only the complete line is consumed (got {first})")
        check(tail.poll() == [], "and polling again does not half-consume the rest")

        # Completing the line releases exactly that record, not a new one.
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('ten"}\n')
        second = tail.poll()
        check([e.payload["node_id"] for e in second] == ["half-written"],
              f"and it arrives once it is whole (got {second})")


def test_bad_records_do_not_end_the_stream() -> None:
    print("\n== one unreadable record must not cost the rest of the run ==")
    with tempfile.TemporaryDirectory() as tmp:
        path = _path(tmp)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("not json at all\n")
            handle.write('["not an object"]\n')
            handle.write('{"kind": "no_such_kind"}\n')
            handle.write(json.dumps({"kind": "node", "node_id": "good"}) + "\n")
        events = ExecutionEventTail(path).poll()
        check([e.payload["node_id"] for e in events] == ["good"],
              f"the one good record survives (got {events})")


def test_nonfinite_is_sanitized_before_the_browser_sees_it() -> None:
    print("\n== a diverged float must not ship as a bare NaN ==")
    with tempfile.TemporaryDirectory() as tmp:
        path = _path(tmp)
        writer = ExecutionEventWriter(path)
        writer.monitor("mon-1", {"loss": float("nan"), "step": 4})
        writer.close()
        raw = path.read_text(encoding="utf-8")
        check("NaN" not in raw, "no bare NaN token in the bytes")
        data = json.loads(raw)["data"]
        check(data["loss"] is None, f"it became null (got {data['loss']!r})")
        check(data["nonfinite"] == {"loss": "nan"},
              f"and is named so a client can say so (got {data.get('nonfinite')})")
        check(data["step"] == 4, "the finite fields of the same frame survive")


def test_missing_and_shrinking_files() -> None:
    print("\n== a file that is not there, and one that shrank ==")
    with tempfile.TemporaryDirectory() as tmp:
        tail = ExecutionEventTail(Path(tmp) / "never-written.jsonl")
        check(tail.poll() == [], "a missing file reads as no records, not an error")

        path = _path(tmp)
        ExecutionEventWriter(path).node({"node_id": "a", "ok": True, "outputs": {},
                                         "error": None, "duration_ms": 1.0})
        tail = ExecutionEventTail(path)
        check(len(tail.poll()) == 1, "one record read")
        check(tail.poll() == [], "offset advanced past it")

        path.write_text("", encoding="utf-8")  # parent already exists
        check(tail.poll() == [], "an emptied file does not raise")
        ExecutionEventWriter(path).node({"node_id": "b", "ok": True, "outputs": {},
                                         "error": None, "duration_ms": 1.0})
        check([e.payload["node_id"] for e in tail.poll()] == ["b"],
              "and the tail picks up from the start again")

def test_unopenable_writer_is_not_fatal() -> None:
    print("\n== an event file that cannot be opened costs reporting, not the run ==")
    blocker = Path(tempfile.mkdtemp(prefix="graph-event-")) / "events.jsonl"
    blocker.mkdir()  # a directory where a file should be: open() fails
    writer = ExecutionEventWriter(blocker)
    check(not writer.available, "the writer reports itself unavailable")
    writer.node({"node_id": "a", "ok": True, "outputs": {}, "error": None,
                 "duration_ms": 1.0})
    writer.outcome(error=None, results_count=1)
    writer.close()
    writer.close()  # idempotent
    check(True, "writing to it raises nothing: the child still runs the graph")


def test_reset_replays_from_the_start() -> None:
    print("\n== reset re-reads, which is how adoption rebuilds history ==")
    with tempfile.TemporaryDirectory() as tmp:
        path = _path(tmp)
        writer = ExecutionEventWriter(path)
        writer.monitor("mon-1", {"step": 1})
        writer.monitor("mon-1", {"step": 2})
        writer.close()

        tail = ExecutionEventTail(path)
        check(len(tail.poll()) == 2, "both reports read once")
        check(tail.poll() == [], "and none on a second poll")
        tail.reset()
        check(len(tail.poll()) == 2,
              "after reset the whole history comes back -- a server that "
              "restarted mid-run must be able to show a dashboard what it missed")


def main() -> None:
    test_round_trip()
    test_torn_tail_is_withheld()
    test_bad_records_do_not_end_the_stream()
    test_nonfinite_is_sanitized_before_the_browser_sees_it()
    test_missing_and_shrinking_files()
    test_unopenable_writer_is_not_fatal()
    test_reset_replays_from_the_start()
    finish()


if __name__ == "__main__":
    main()