"""The event wire contract: generated schema, real frames, and the frontend.

Three separate things, because they fail differently:

1. **The schema matches reality.** Every event is constructed, serialised
   by the real serializer, and validated against its own generated
   schema. Catches a dataclass gaining a field the schema does not know,
   or the serializer adding a key the schema rejects.

2. **The validator is right.** Cross-checked against the real
   ``jsonschema`` when it is importable, so the hand-written subset is
   verified rather than assumed. Skipped with a notice otherwise.

3. **The frontend only reads fields that exist.** The guard this whole
   exercise exists for (`docs/design/backend/09-event-contract.md`):
   a backend rename stops every frame carrying the field, the frontend
   reads `undefined`, and the page renders an em dash with no error on
   either side and coverage green on both.

   What (3) does and does not catch, since a regex over JavaScript is not
   a parser:

   * It collects field reads **per variable**, and only from variables
     that provably are frames -- those compared `.type` against a wire
     type name. That is what keeps `opt.type === "checkbox"` and
     `input.type` out of the results.
   * It checks against the **union** of every event's fields, not per
     event type. So a field renamed on one event but still present on
     another would pass. That is a real gap and it is not worth closing
     with brace-matching regex: the bug this must catch is a name that
     exists *nowhere*, which the union catches exactly.

Run directly: python backend/tests/test_event_contract.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.domain.events import (  # noqa: E402  (path set up above)
    GraphExecutionFailed,
    GraphExecutionFinished,
    GraphExecutionProgressed,
    GraphExecutionQueued,
    GraphExecutionStarted,
    GraphExecutionStopped,
    GraphExecutionsDeleted,
)
from backend.presentation.event_schema import (  # noqa: E402
    SYNTHETIC_TYPES,
    all_event_types,
    schema_for,
    validate,
    wire_field_names,
)
from backend.presentation.sse import serialize_event  # noqa: E402
from backend.tests.support import check, finish  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_JS = REPO_ROOT / "frontend" / "js"
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)

#: Every event class, imported explicitly. ``all_event_types()`` walks
#: ``__subclasses__``, which only sees loaded modules, so an event left
#: out of this list would silently vanish from the contract instead of
#: failing.
ALL_EVENT_CLASSES = [
    GraphExecutionQueued, GraphExecutionStarted, GraphExecutionProgressed,
    GraphExecutionFinished, GraphExecutionFailed, GraphExecutionStopped,
    GraphExecutionsDeleted,
]


def _sample(annotation: Any) -> Any:
    """A valid Python value for one field annotation.

    Derived from the annotations rather than written out per event, so
    adding an event does not mean adding a hand-built instance for it.
    """
    if annotation is datetime:
        return NOW
    if get_origin(annotation) is Union:
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        return _sample(non_none[0])
    return {int: 3, str: "x", float: 1.5, bool: True}[annotation]


def _instance(cls: type) -> Any:
    hints = get_type_hints(cls)
    kwargs = {
        name: _sample(hints[name])
        for name in cls.__dataclass_fields__
        if name != "occurred_at"
    }
    return cls(occurred_at=NOW, **kwargs)


# --------------------------------------------------------------------------
# 1. the schema describes what the serializer actually emits
# --------------------------------------------------------------------------

def test_schema_covers_every_event() -> None:
    print("\n== every event is in the generated contract ==")
    types = all_event_types()
    check(len(ALL_EVENT_CLASSES) == 7,
          f"the import list has all 7 event classes (got "
          f"{len(ALL_EVENT_CLASSES)}) -- the 7 run events went with the "
          f"supervised-subprocess route")
    for cls in ALL_EVENT_CLASSES:
        check(cls.wire_name() in types,
              f"{cls.__name__} -> {cls.wire_name()} is in the contract")
    check("stream_opened" in types, "the synthetic opening frame is declared too")
    check(len(types) == len(ALL_EVENT_CLASSES) + len(SYNTHETIC_TYPES),
          f"and nothing else is (got {len(types)} types, "
          f"{len(ALL_EVENT_CLASSES)} events + {len(SYNTHETIC_TYPES)} synthetic)")

    # The classmethod and the property must not drift: the generator uses
    # one and the whole codebase uses the other.
    for cls in ALL_EVENT_CLASSES:
        check(_instance(cls).event_type == cls.wire_name(),
              f"{cls.__name__}: event_type agrees with wire_name")


def test_real_frames_validate() -> None:
    print("\n== a real serialised frame validates against its own schema ==")
    for cls in ALL_EVENT_CLASSES:
        event = _instance(cls)
        name = event.event_type
        payload = json.loads(serialize_event(event, seq=7))
        errors = validate(payload, schema_for(name))
        check(not errors,
              f"{name}: a frame the real serializer produced is valid "
              f"(got {errors})")
        check(payload["seq"] == 7 and payload["type"] == name,
              f"{name}: carries its seq and type")


def test_nullable_and_nonfinite() -> None:
    print("\n== nulls and non-finite floats are still valid frames ==")
    # None everywhere it is allowed.
    finished = GraphExecutionFinished(occurred_at=NOW, execution_id=1, nodes=0)
    check(not validate(json.loads(serialize_event(finished, seq=1)),
                       schema_for("graph_execution_finished")),
          "a minimal GraphExecutionFinished validates")

    failed = GraphExecutionFailed(occurred_at=NOW, execution_id=1, error=None)
    check(not validate(json.loads(serialize_event(failed, seq=2)),
                       schema_for("graph_execution_failed")),
          "graph_execution_failed with a null error validates -- the anyOf works")

    # The diverged-loss case is a real payload, not a hypothetical: a NaN
    # becomes null plus a `nonfinite` sibling, and the frontend reads that
    # sibling to render an error instead of a blank (docs 07 F-03).
    diverged = GraphExecutionProgressed(
        occurred_at=NOW, execution_id=1, node_id="nA", ok=True,
        duration_ms=float("nan"),
    )
    payload = json.loads(serialize_event(diverged, seq=3))
    check(payload["duration_ms"] is None, "the NaN became null on the wire")
    check(payload.get("nonfinite") == {"duration_ms": "nan"},
          f"and it is named in `nonfinite` (got {payload.get('nonfinite')})")
    check(not validate(payload, schema_for("graph_execution_progressed")),
          f"a diverged frame still validates (got "
          f"{validate(payload, schema_for('graph_execution_progressed'))})")

    # The schema must also REJECT a wrong shape, or it is not a contract.
    check(validate({**payload, "duration_ms": "not a number"},
                   schema_for("graph_execution_progressed")),
          "a string where the schema says number is rejected")
    check(validate({"type": "graph_execution_progressed"},
                   schema_for("graph_execution_progressed")),
          "a frame missing its required fields is rejected")
    check(validate({**payload, "type": "graph_execution_finished"},
                   schema_for("graph_execution_progressed")),
          "a frame whose type does not match its schema is rejected")


def test_synthetic_frame_validates() -> None:
    print("\n== the frames with no dataclass behind them ==")
    opened = {
        "type": "stream_opened",
        "occurred_at": NOW.isoformat(),
        "resync_required": True,
        "replayed_through": None,
    }
    check(not validate(opened, schema_for("stream_opened")),
          f"stream_opened validates (got {validate(opened, schema_for('stream_opened'))})")
    check(not validate({**opened, "replayed_through": 42},
                       schema_for("stream_opened")),
          "and with a replay watermark")
    check(validate({"type": "stream_opened"}, schema_for("stream_opened")),
          "one missing resync_required is rejected -- the field the "
          "frontend branches on")


# --------------------------------------------------------------------------
# 2. the hand-written validator agrees with a real one
# --------------------------------------------------------------------------

def test_validator_agrees_with_jsonschema() -> None:
    print("\n== the validator agrees with jsonschema, where it is installed ==")
    try:
        import jsonschema
    except ImportError:
        print("  (jsonschema not installed -- hand-written validator only; "
              "it is not a dependency of this project)")
        return

    schemas = {name: schema_for(name) for name in all_event_types()}
    validator = jsonschema.Draft202012Validator
    checked = 0
    disagreements = []
    for cls in ALL_EVENT_CLASSES:
        event = _instance(cls)
        name = event.event_type
        cases = [
            json.loads(serialize_event(event, seq=1)),
            json.loads(serialize_event(event, seq=1)),
            json.loads(serialize_event(event)),  # no seq, as a synthetic frame
            {**json.loads(serialize_event(event, seq=1)), "duration_ms": "x"},
            {"type": name},
        ]
        for payload in cases:
            mine = bool(validate(payload, schemas[name]))
            theirs = not validator(schemas[name]).is_valid(payload)
            checked += 1
            if mine != theirs:
                disagreements.append(
                    f"{name} {payload!r}: mine={mine} jsonschema={theirs}"
                )
    check(not disagreements,
          f"{checked} payload/schema pairs agreed (got "
          f"{disagreements[:2]})")


# --------------------------------------------------------------------------
# 3. the frontend reads only fields that exist
# --------------------------------------------------------------------------

_FRAME_TYPE_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\.type\s*===?\s*[\"']([a-z_]+)[\"']")
_FIELD_RE = re.compile(r"\b([A-Za-z_$][\w$]*)\.([a-z_][a-z0-9_]*)\b")


def _frame_variables(text: str, wire_types: set[str]) -> set[str]:
    """Variables the file compares `.type` against a wire type name."""
    return {
        name for name, literal in _FRAME_TYPE_RE.findall(text)
        if literal in wire_types
    }


def _field_reads(text: str, variables: set[str]) -> set[str]:
    return {
        field for name, field in _FIELD_RE.findall(text)
        if name in variables
    }


def test_frontend_reads_only_declared_fields() -> None:
    print("\n== the frontend reads no field the contract does not declare ==")
    wire_types = set(all_event_types())
    allowed = wire_field_names()

    files = sorted(FRONTEND_JS.rglob("*.js"))
    check(len(files) >= 8, f"scanned the real frontend tree ({len(files)} files)")

    scanned_frames = 0
    offenders: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        variables = _frame_variables(text, wire_types)
        if not variables:
            continue
        scanned_frames += 1
        rel = path.relative_to(REPO_ROOT)
        for field in sorted(_field_reads(text, variables) - allowed):
            offenders.append(f"{rel}: {sorted(variables)}.{field}")

    check(scanned_frames >= 2,
          f"{scanned_frames} files handle frames, so the scan has something "
          f"to say (a scan that finds nothing may just be broken). This was 3 "
          f"before the run dashboard went away.")
    check(not offenders,
          f"every field read on a frame is declared by the contract "
          f"(undeclared: {offenders})")


def test_the_scanner_can_actually_see_a_violation() -> None:
    print("\n== the scan would notice one ==")
    wire_types = set(all_event_types())
    sample = '''
      function handleEvent(e) {
        if (e.type === "graph_execution_finished") {
          render(e.execution_id, e.nodes, e.definitely_not_a_field);
        }
      }
    '''
    variables = _frame_variables(sample, wire_types)
    check(variables == {"e"}, f"the frame variable is identified (got {variables})")
    reads = _field_reads(sample, variables)
    check("definitely_not_a_field" in reads,
          f"and the invented field is among the reads (got {sorted(reads)})")
    check(bool(reads - wire_field_names()),
          "so it is not in the contract, and the check would fail")

    # ...and it does not fire on ordinary JavaScript. The whole defence
    # is the identification step: `opt.type` here is a real `.type`, it is
    # just not compared against a wire type name, so `opt` never becomes a
    # frame variable and its reads are never collected at all.
    noise = '''
      const opt = { type: "checkbox" };
      if (opt.type === "checkbox") { input.checked = true; }
      input.type = "number";
      const p = fmt.duration(1200);
    '''
    identified = _frame_variables(noise, wire_types)
    check(not identified,
          f"a config view's `opt.type` is not mistaken for a frame (got "
          f"{identified})")
    check(not _field_reads(noise, identified),
          "so none of its field reads reach the contract check")


def main() -> None:
    test_schema_covers_every_event()
    test_real_frames_validate()
    test_nullable_and_nonfinite()
    test_synthetic_frame_validates()
    test_validator_agrees_with_jsonschema()
    test_frontend_reads_only_declared_fields()
    test_the_scanner_can_actually_see_a_violation()
    finish()


if __name__ == "__main__":
    main()