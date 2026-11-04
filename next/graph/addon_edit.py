"""Changing a graph's addons without an editor: enable, configure, disable, remove, and loading an addon-set file.

Every function takes a ``GraphDoc`` and returns a new one, or raises ``AddonError`` with the validator's messages when the
result would break an addon rule. The editor, the command line and scripts use the same calls (rework/31-node-addons.md, 6b).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .addons import ADDON_ID, GRAPH, AddonError, AddonSpec, slots_of
from .document import GraphDoc, NodeSpec
from .registry import NodeRegistry
from .validate import ERROR, validate

TOUCHED = ("addon_", "unknown_param", "type_mismatch", "out_of_range", "invalid_choice", "missing_required_input")


def host_of(doc: GraphDoc, ref: str) -> str:
    """The address of the host a reference names: ``graph``, a node uid, a node title or a class name used once."""
    if ref == GRAPH:
        return GRAPH
    for key in ("uid", "title", "class_name"):
        found = [n.uid for n in doc.nodes if getattr(n, key) == ref and ref]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise AddonError(f"{ref!r} names {len(found)} nodes; use a uid or a title")
    raise AddonError(f"no node or graph called {ref!r}")


def available(doc: GraphDoc, registry: NodeRegistry, ref: str) -> list[dict]:
    """The addons a host's slots accept: kind, slot and display name."""
    host = host_of(doc, ref)
    cls, label = _class_and_label(doc, registry, host)
    slots = {s.name for s in slots_of(cls)}
    return [{"kind": kind, "slot": a.SLOT, "name": a.DISPLAY_NAME or kind}
            for kind, a in sorted(registry.addons().items()) if a.SLOT in slots and (not a.FITS or label in a.FITS)]


def enable(doc: GraphDoc, registry: NodeRegistry, ref: str, kind: str, params: dict | None = None, *,
           addon_id: str | None = None, enabled: bool = True) -> GraphDoc:
    host = host_of(doc, ref)
    out, new_id = _add(doc, host, kind, params or {}, addon_id, enabled)
    _check(out, registry, {host: {new_id}})
    return out


def configure(doc: GraphDoc, registry: NodeRegistry, ref: str, addon_id: str, **params: Any) -> GraphDoc:
    host = host_of(doc, ref)
    out = _change(doc, host, addon_id, lambda a: replace(a, params={**a.params, **params}))
    _check(out, registry, {host: {addon_id}})
    return out


def disable(doc: GraphDoc, registry: NodeRegistry, ref: str, addon_id: str) -> GraphDoc:
    """Switch an addon off; its parameters stay in the file."""
    host = host_of(doc, ref)
    out = _change(doc, host, addon_id, lambda a: replace(a, enabled=False))
    _check(out, registry, {host: {addon_id}})
    return out


def remove(doc: GraphDoc, registry: NodeRegistry, ref: str, addon_id: str) -> GraphDoc:
    host = host_of(doc, ref)
    if not _has(doc, host, addon_id):
        _missing(host, addon_id)
    out = _replace_specs(doc, host, tuple(a for a in _specs(doc, host) if a.id != addon_id))
    _check(out, registry, {host: set()})
    return out


def apply_set(doc: GraphDoc, registry: NodeRegistry, data: dict) -> GraphDoc:
    """Merge an addon-set (``{"trainer": [{"kind": ..., "params": {...}}], "graph": [...]}``) into a graph."""
    if not isinstance(data, dict):
        raise AddonError("an addon set is an object mapping a node (uid, title, class) or 'graph' to a list of addons")
    touched: dict[str, set[str]] = {}
    for ref, entries in data.items():
        host = host_of(doc, ref)
        if not isinstance(entries, list):
            raise AddonError(f"{ref}: the addons must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("kind"), str):
                raise AddonError(f"{ref}: every addon needs a 'kind'")
            doc, new_id = _add(doc, host, entry["kind"], dict(entry.get("params", {})), entry.get("id"), entry.get("enabled", True))
            touched.setdefault(host, set()).add(new_id)
    _check(doc, registry, touched)
    return doc


def load_set(doc: GraphDoc, registry: NodeRegistry, path: str | Path) -> GraphDoc:
    return apply_set(doc, registry, json.loads(Path(path).read_text(encoding="utf8")))


# -- helpers ---------------------------------------------------------------

def _specs(doc: GraphDoc, host: str) -> tuple[AddonSpec, ...]:
    return doc.addons if host == GRAPH else doc.node(host).addons


def _has(doc: GraphDoc, host: str, addon_id: str) -> bool:
    return any(a.id == addon_id for a in _specs(doc, host))


def _missing(host: str, addon_id: str):
    raise AddonError(f"{host} has no addon {addon_id!r}")


def _replace_specs(doc: GraphDoc, host: str, specs: tuple[AddonSpec, ...]) -> GraphDoc:
    if host == GRAPH:
        return replace(doc, addons=specs)
    nodes = tuple(replace(n, addons=specs) if n.uid == host else n for n in doc.nodes)
    return replace(doc, nodes=nodes)


def _change(doc: GraphDoc, host: str, addon_id: str, how) -> GraphDoc:
    if not _has(doc, host, addon_id):
        _missing(host, addon_id)
    return _replace_specs(doc, host, tuple(how(a) if a.id == addon_id else a for a in _specs(doc, host)))


def _add(doc: GraphDoc, host: str, kind: str, params: dict, addon_id: str | None, enabled: bool) -> tuple[GraphDoc, str]:
    specs = _specs(doc, host)
    taken = {a.id for a in specs}
    if addon_id is None:
        n = 1
        while f"a{n}" in taken:
            n += 1
        addon_id = f"a{n}"
    elif not ADDON_ID.match(addon_id) or addon_id in taken:
        raise AddonError(f"addon id {addon_id!r} is not usable on {host} (letters, digits and _, and not taken)")
    return _replace_specs(doc, host, (*specs, AddonSpec(addon_id, kind, enabled, dict(params)))), addon_id


def _class_and_label(doc: GraphDoc, registry: NodeRegistry, host: str):
    if host == GRAPH:
        return None, GRAPH
    spec: NodeSpec = doc.node(host)
    return registry.load()[0].get(spec.class_name), spec.class_name


def _check(doc: GraphDoc, registry: NodeRegistry, touched: dict[str, set[str]]) -> None:
    """Refuse when the validator finds addon problems on a changed host (problems elsewhere in the graph are not this call's)."""
    problems = []
    for issue in validate(doc, registry).issues:
        if issue.severity != ERROR or issue.uid not in touched or not issue.code.startswith(TOUCHED):
            continue
        ours = issue.code.startswith("addon_") or (issue.port or "").partition(".")[0] in touched[issue.uid]
        if ours:
            problems.append(issue.message)
    if problems:
        raise AddonError("; ".join(problems))
