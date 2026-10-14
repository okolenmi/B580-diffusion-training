"""Serialising node classes for the editor (rework/contracts/catalog.md). Never instantiates a node."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from .node import Node, is_field
from .ports import EventIn, EventOut, Port, type_mro_names, type_name
from .registry import NodeRegistry
from .settings import SettingsRegistry

CATALOG_FORMAT = 1


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    return repr(value)[:80]


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


def port_info(port: Port) -> dict:
    return {
        "name": port.name, "label": port.label or _label(port.name), "type": type_name(port.type),
        "type_mro": type_mro_names(port.type), "required": port.required, "default": _json_value(port.default),
        "doc": port.doc, "group": port.group, "advanced": port.advanced, "unit": port.unit,
        "minimum": port.minimum, "maximum": port.maximum, "step": port.step,
        "choices": list(port.choices) if port.choices else None, "picker": port.picker,
        "visible_when": [port.visible_when[0], _json_value(port.visible_when[1])] if port.visible_when else None,
        "widget_only": port.widget_only, "multi": port.multi, "field": is_field(port),
        "ui": _json_value(dict(port.ui)),
    }


def event_port_info(name: str, port: EventIn | EventOut) -> dict:
    return {"name": name, "label": port.label or _label(name), "doc": port.doc,
            "payload": type_name(port.payload), "payload_mro": type_mro_names(port.payload)}


def _auto_display_name(class_name: str) -> str:
    base = class_name[:-4] if class_name.endswith("Node") else class_name
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", base)


def has_diagnostics(cls: type[Node]) -> bool:
    return cls.diagnostics is not Node.diagnostics


def node_info(cls: type[Node]) -> dict:
    return {
        "class_name": cls.__name__,
        "display_name": cls.DISPLAY_NAME or _auto_display_name(cls.__name__),
        "category": cls.CATEGORY,
        "doc": (cls.__doc__ or "").strip().split("\n\n")[0],
        "inputs": [port_info(p) for p in cls.INPUTS.values()],
        "outputs": [port_info(p) for p in cls.OUTPUTS.values()],
        "groups": [{"id": g.id, "label": g.label, "collapsed": g.collapsed, "advanced": g.advanced,
                    "visible_when": [g.visible_when[0], _json_value(g.visible_when[1])] if g.visible_when else None,
                    "doc": g.doc} for g in cls.GROUPS],
        "events_in": [event_port_info(n, p) for n, p in cls.EVENTS_IN.items()],
        "events_out": [event_port_info(n, p) for n, p in cls.EVENTS_OUT.items()],
        "modules": [m.as_dict() for m in cls.MODULES],
        "reads_settings": list(cls.READS_SETTINGS),
        "settings": [s.as_dict() for s in cls.SETTINGS],
        "dynamic": ({"depends_on": list(cls.DYNAMIC.depends_on), "debounce_ms": cls.DYNAMIC.debounce_ms}
                    if cls.DYNAMIC else None),
        "has_diagnostics": has_diagnostics(cls),
        "accepts_unfinished": cls.ACCEPTS_UNFINISHED,
    }


def handle_types(classes: Iterable[type[Node]]) -> dict[str, list[str]]:
    seen: dict[str, list[str]] = {}
    def add(t: Any) -> None:
        if isinstance(t, type):
            seen.setdefault(t.__name__, type_mro_names(t))
    for cls in classes:
        for port in (*cls.INPUTS.values(), *cls.OUTPUTS.values()):
            add(port.type)
        for ev in (*cls.EVENTS_IN.values(), *cls.EVENTS_OUT.values()):
            add(ev.payload)
    return seen


def build_catalog(registry: NodeRegistry, settings: SettingsRegistry | None = None,
                  families: list[dict] | None = None) -> dict:
    classes, errors = registry.load()
    settings = settings or registry.settings_registry()
    ordered = sorted(classes.values(), key=lambda c: (c.CATEGORY, c.__name__))
    return {
        "catalog_format": CATALOG_FORMAT,
        "nodes": [node_info(c) for c in ordered],
        "settings": [s.as_dict() for s in settings.specs()],
        "handle_types": handle_types(ordered),
        "families": families or [],
        "load_errors": list(errors) + [{"module": "settings", "error": e} for e in settings.errors],
    }
