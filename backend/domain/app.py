"""App documents -- a published, frozen page over a graph.

An App is what makes a workflow usable by someone who should not have to meet the graph. It holds a **snapshot of the graph** and a
**root container** of cards (``domain/board.py``) that point at the snapshot's modules. The graph editor stays the place graphs are
authored, and keeps its own live views; an App is the explicit, versioned copy: publishing again with a changed graph makes the next
version, and a changed layout alone does not.

Using an App changes only the fields its cards expose (``application/app_binding.py``); everything else in the snapshot is what the
author saved. A card whose target vanished from a newer snapshot stays visible as an orphan rather than making the App unloadable.

Pure functions over dicts, no I/O. The document is stored verbatim apart from the format stamp and filled defaults.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .board import MAX_NODE_ID, normalize_container, validate_container

APP_FORMAT = 1

ACCENTS: tuple[str, ...] = ("blue", "teal", "green", "amber", "red", "violet")
DENSITIES: tuple[str, ...] = ("comfortable", "compact")

#: What a run reads from a graph; everything else (layout, views, titles) is presentation and does not make a new version.
_NODE_KEYS = ("uid", "class_name", "params", "addons")
_GRAPH_KEYS = ("settings", "edges", "event_edges", "addons")


def validate_app_doc(doc: Any) -> list[str]:
    """Every shape problem as ``"path: message"`` (empty = well-formed)."""
    if not isinstance(doc, dict):
        return ["app: must be an object"]
    problems: list[str] = []
    _validate_graph(doc.get("graph"), problems)
    source = doc.get("source_graph")
    if source is not None and not isinstance(source, str):
        problems.append("source_graph: must be a string or null")
    _validate_style(doc.get("style"), problems)
    root = doc.get("root")
    if root is None:
        problems.append("root: an app needs its page (a container)")
    else:
        validate_container(root, "root", problems)
    return problems


def _validate_graph(graph: Any, problems: list[str]) -> None:
    if graph is None:
        problems.append("graph: an app needs the graph snapshot it runs")
        return
    if not isinstance(graph, dict):
        problems.append("graph: must be an object")
        return
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        problems.append("graph.nodes: must be a list")
    else:
        for index, node in enumerate(nodes):
            uid = node.get("uid") if isinstance(node, dict) else None
            if not isinstance(uid, str) or not uid or len(uid) > MAX_NODE_ID:
                problems.append(f"graph.nodes[{index}]: must be an object with a uid")
    for key in ("edges", "event_edges"):
        if not isinstance(graph.get(key, []), list):
            problems.append(f"graph.{key}: must be a list")


def _validate_style(style: Any, problems: list[str]) -> None:
    if style is None:
        return
    if not isinstance(style, dict):
        problems.append("style: must be an object")
        return
    accent, density = style.get("accent"), style.get("density")
    if accent is not None and accent not in ACCENTS:
        problems.append(f"style.accent: {accent!r} is not one of {list(ACCENTS)}")
    if density is not None and density not in DENSITIES:
        problems.append(f"style.density: {density!r} is not one of {list(DENSITIES)}")


def normalize_app_doc(doc: dict) -> dict:
    """The stored shape: format stamped, defaults filled, unknown keys kept. Call only on a document ``validate_app_doc`` accepted."""
    out = dict(doc)
    out["format"] = APP_FORMAT
    out["source_graph"] = doc.get("source_graph")
    out["style"] = {"accent": "blue", "density": "comfortable", **(doc.get("style") or {})}
    out["root"] = normalize_container(doc["root"])
    return out


def graph_hash(graph: dict) -> str:
    """Identity of what a run would do with this snapshot: nodes by uid (class, params, addons), the wires, the settings. Node
    order and layout do not count; edge order does (it is the argument order of a multi port)."""
    nodes = sorted(
        ({key: node.get(key) for key in _NODE_KEYS} for node in graph.get("nodes", [])), key=lambda n: n["uid"])
    canonical = {"nodes": nodes, **{key: graph.get(key) for key in _GRAPH_KEYS}}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def diff_graphs(old: dict, new: dict) -> dict:
    """What publishing ``new`` over ``old`` would change, by node uid, for the "update from graph" review."""
    before = {node["uid"]: node for node in old.get("nodes", [])}
    after = {node["uid"]: node for node in new.get("nodes", [])}
    shared = before.keys() & after.keys()
    return {
        "added": sorted(after.keys() - before.keys()),
        "removed": sorted(before.keys() - after.keys()),
        "changed": sorted(uid for uid in shared if {k: before[uid].get(k) for k in _NODE_KEYS}
                          != {k: after[uid].get(k) for k in _NODE_KEYS}),
        "wiring_changed": any(old.get(key) != new.get(key) for key in ("edges", "event_edges")),
        "settings_changed": old.get("settings") != new.get("settings"),
    }
