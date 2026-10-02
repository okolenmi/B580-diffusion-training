"""The event wire contract, as a JSON Schema generated from the dataclasses.

Generated rather than written, because a hand-maintained schema is a
second description of the same thing and drifts from the first within a
release -- which is the bug this exists to prevent. The failure it guards
against has no other guard at all: the backend renames a field, every
frame stops carrying it, and the frontend reads `undefined` and renders an
em dash, with no error on either side and coverage green on both
(`docs/design/backend/09-event-contract.md`).

What is on the wire is **not** just the dataclass
----------------------------------------------------
The schema describes the frame a client actually receives, which is the
dataclass's own fields plus three keys no dataclass declares:

``type``
    Added by ``sse.serialize_event`` from ``DomainEvent.event_type``.
``seq``
    The stream position, added by the same function (docs 09).
``nonfinite``
    Added by ``json_safe.sanitize`` to any payload holding a non-finite
    float -- ``null`` plus ``{key: "nan" | "inf" | "-inf"}`` (docs 07 F-03).
    It is conditional: a healthy frame does not carry it.

Describing only the dataclasses would therefore have produced a schema
that rejects a real frame, and ``frontend/js/views/run.js`` reads
``e.nonfinite`` specifically so that a diverged loss renders as an error
rather than as a blank.

Generated here, in ``presentation/``, rather than in ``domain/``, because
those three keys are this layer's doing. The dataclasses stay ignorant of
JSON, which is the point of them.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Any, Union, get_args, get_origin, get_type_hints

from ..domain.events import DomainEvent

#: Synthetic frames the server sends that are not bus events. They have no
#: dataclass, so the generator cannot find them, and the frontend reads
#: their fields -- so they are declared here and pinned by a test.
SYNTHETIC_TYPES: dict[str, tuple[str, ...]] = {
    "stream_opened": ("resync_required", "replayed_through"),
}

#: Envelope keys added to every event frame by the serializer.
ENVELOPE_KEYS: tuple[str, ...] = ("type", "seq", "nonfinite")

#: Python type -> (JSON Schema type, or None for a union).
_SCALARS: dict[Any, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    datetime: "string",  # ISO-8601 on the wire
}


def all_event_types() -> tuple[str, ...]:
    """Every event type the server can emit, sorted.

    Uses ``__subclasses__``, which only sees modules that are *loaded* --
    so ``events`` must be imported for this to be complete. It is, above;
    the caveat is recorded because it is the kind of thing that silently
    shrinks a result later (``test_error_contract.py`` hits the same one).
    """
    names = [sub.wire_name() for sub in _event_classes()]
    return tuple(sorted(names + list(SYNTHETIC_TYPES)))


def _event_classes() -> list[type[DomainEvent]]:
    return [
        sub for sub in DomainEvent.__subclasses__()
        if is_dataclass(sub) and not inspect.isabstract(sub)
    ]


def event_class_for(event_type: str) -> type[DomainEvent] | None:
    for cls in _event_classes():
        if cls.wire_name() == event_type:
            return cls
    return None


def _json_type(annotation: Any) -> dict:
    """One annotation to a JSON Schema fragment.

    Handles exactly what ``domain/events.py`` uses -- scalars, ``X | None``
    and nothing else. An unknown construct raises rather than passing
    silently: a schema that quietly describes the wrong thing is the
    failure this whole module is about.
    """
    if annotation in _SCALARS:
        kind = _SCALARS[annotation]
        if annotation is float:
            # A float must be nullable on the wire even when the
            # annotation is not `float | None`. `json_safe.sanitize`
            # replaces any non-finite float with null and names it in
            # `nonfinite` (docs 07 F-03) -- so a plain `float` field
            # legitimately arrives as null on exactly the frame a user
            # most wants to see, the diverged one. Making it nullable only
            # for floats is not a concession: ints cannot be NaN, so
            # nothing ever nulls them.
            return {"anyOf": [{"type": kind}, {"type": "null"}]}
        return {"type": kind}

    # X | None -- PEP 604 at runtime is typing.Union.
    if get_origin(annotation) is Union:
        options = get_args(annotation)
        non_none = [arg for arg in options if arg is not type(None)]
        nullable = len(non_none) != len(options)
        if len(non_none) != 1:
            raise TypeError(
                f"event field annotated {annotation!r}: only `T` and "
                f"`T | None` are supported, not unions of several types"
            )
        fragment = _json_type(non_none[0])
        if nullable:
            fragment = {"anyOf": [fragment, {"type": "null"}]}
        return fragment

    raise TypeError(
        f"event field annotated {annotation!r}: no JSON Schema mapping. "
        f"Add one to _SCALARS or to the union handling rather than letting "
        f"the generated schema describe the wrong type."
    )


def schema_for(event_type: str) -> dict:
    """The JSON Schema for one wire frame, or for a synthetic one."""
    if event_type in SYNTHETIC_TYPES:
        props: dict[str, dict] = {
            "type": {"type": "string", "const": event_type},
            "occurred_at": {"type": "string"},
            "resync_required": {"type": "boolean"},
            "replayed_through": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        }
        required = ["type", "occurred_at", "resync_required"]
        return _envelope(event_type, props, required)

    cls = event_class_for(event_type)
    if cls is None:
        raise KeyError(
            f"no event class produces the wire type {event_type!r}"
        )
    # get_type_hints, not __annotations__: this module tree uses
    # `from __future__ import annotations`, so the raw annotations are
    # strings. Resolving them also collapses `RunId`, which is an alias
    # for int rather than a distinct type.
    hints = get_type_hints(cls)
    props = {name: _json_type(hints[name]) for name in _field_names(cls)}
    required = sorted(props)
    return _envelope(event_type, props, required)


def _field_names(cls: type[DomainEvent]) -> list[str]:
    """Own fields first, then inherited -- ``fields()`` order, minus the
    base's ``occurred_at``, which the envelope adds itself."""
    return [f.name for f in fields(cls) if f.name != "occurred_at"]


def _envelope(event_type: str, props: dict, required: list[str]) -> dict:
    props["type"] = {"type": "string", "const": event_type}
    props["occurred_at"] = {"type": "string"}
    if event_type not in SYNTHETIC_TYPES:
        props["seq"] = {"type": "integer"}
        props["nonfinite"] = {"type": "object"}
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": event_type,
        "type": "object",
        "properties": props,
        "required": sorted(set(required) | {"type", "occurred_at"}),
    }
    if event_type not in SYNTHETIC_TYPES:
        schema["required"] = sorted(set(schema["required"]) | {"seq"})
    return schema


def all_schemas() -> dict[str, dict]:
    return {name: schema_for(name) for name in all_event_types()}


def wire_field_names() -> frozenset[str]:
    """Every field name any frame can carry.

    The frontend check compares against this union rather than per event
    type; the reason it can get away with that is recorded in
    ``tests/test_event_contract.py``, which pins the per-type accuracy
    separately.
    """
    names: set[str] = set(ENVELOPE_KEYS) | {"occurred_at"}
    for schema in all_schemas().values():
        names.update(schema["properties"])
    return frozenset(names)


# --------------------------------------------------------------------------
# A deliberately small validator
# --------------------------------------------------------------------------

def validate(payload: Any, schema: dict, *, path: str = "frame") -> list[str]:
    """Errors in ``payload`` against ``schema``; empty means valid.

    Hand-written because ``jsonschema`` is not a dependency of this
    project -- it is present here only as a transitive dependency of
    ComfyUI's ``matrix-nio``, which is not a promise. It covers exactly
    the constructs the generator above emits, and **raises** on anything
    else rather than passing it: a validator that quietly ignores what it
    does not understand is worse than no validator, because it reports
    green.

    ``tests/test_event_contract.py`` cross-checks this against the real
    ``jsonschema`` whenever it happens to be installed, so the subset
    below is verified rather than merely intended.
    """
    errors: list[str] = []

    if "anyOf" in schema:
        if not any(not validate(payload, option, path=path)
                   for option in schema["anyOf"]):
            errors.append(f"{path}: {payload!r} matches none of anyOf")
        return errors

    if "const" in schema:
        if payload != schema["const"]:
            errors.append(f"{path}: {payload!r} != const {schema['const']!r}")
        return errors

    expected = schema.get("type")
    if expected and not _is_type(payload, expected):
        errors.append(f"{path}: {payload!r} is not a JSON {expected}")
        return errors

    if expected == "object":
        for name in schema.get("required", []):
            if name not in payload:
                errors.append(f"{path}: missing required {name!r}")
        for name, sub in schema.get("properties", {}).items():
            if name in payload:
                errors.extend(validate(payload[name], sub, path=f"{path}.{name}"))

    return errors


def _is_type(value: Any, expected: str) -> bool:
    # bool before int: isinstance(True, int) is True, and a JSON boolean is
    # not an integer. This ordering is the whole reason it is written out.
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    raise TypeError(f"validator does not know the JSON type {expected!r}")