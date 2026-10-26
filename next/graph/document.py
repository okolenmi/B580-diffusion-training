"""The saved graph document (rework/contracts/graph-json.md)."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterable

FORMAT = 3
UID_PATTERN = re.compile(r"^n_[a-z2-7]{6}$")


class GraphFormatError(ValueError):
    """The document is not a graph this code can read (wrong shape or format number)."""


def new_uid(existing: Iterable[str] = ()) -> str:
    taken = set(existing)
    while True:
        uid = "n_" + base64.b32encode(os.urandom(4)).decode("ascii").lower()[:6]
        if uid not in taken:
            return uid


@dataclass(frozen=True)
class NodeSpec:
    uid: str
    class_name: str
    params: dict = field(default_factory=dict)
    title: str = ""
    layout: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"uid": self.uid, "title": self.title, "class_name": self.class_name,
                "params": self.params, "layout": self.layout}


@dataclass(frozen=True)
class Edge:
    from_node: str
    from_port: str
    to_node: str
    to_port: str

    def as_dict(self) -> dict:
        return {"from_node": self.from_node, "from_port": self.from_port,
                "to_node": self.to_node, "to_port": self.to_port}


@dataclass(frozen=True)
class GraphDoc:
    nodes: tuple[NodeSpec, ...] = ()
    edges: tuple[Edge, ...] = ()
    event_edges: tuple[Edge, ...] = ()
    settings: dict = field(default_factory=dict)
    views: tuple = ()
    extras: dict = field(default_factory=dict)

    def node(self, uid: str) -> NodeSpec | None:
        for spec in self.nodes:
            if spec.uid == uid:
                return spec
        return None

    @property
    def uids(self) -> list[str]:
        return [n.uid for n in self.nodes]

    def edges_to(self, uid: str) -> list[Edge]:
        return [e for e in self.edges if e.to_node == uid]

    def event_edges_from(self, uid: str) -> list[Edge]:
        return [e for e in self.event_edges if e.from_node == uid]

    def for_run(self) -> GraphDoc:
        """The document without presentation: layouts and views never reach a run."""
        nodes = tuple(NodeSpec(n.uid, n.class_name, n.params, n.title, {}) for n in self.nodes)
        return GraphDoc(nodes, self.edges, self.event_edges, self.settings, (), self.extras)

    def as_dict(self) -> dict:
        out = dict(self.extras)
        out.update({
            "format": FORMAT,
            "settings": dict(self.settings),
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "event_edges": [e.as_dict() for e in self.event_edges],
            "views": list(self.views),
        })
        return out


_KNOWN_KEYS = {"format", "settings", "nodes", "edges", "event_edges", "views"}


def _require(value: Any, kind: type, where: str):
    if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
        raise GraphFormatError(f"{where} must be a {kind.__name__}")
    return value


def _edge(raw: Any, where: str) -> Edge:
    _require(raw, dict, where)
    values = []
    for key in ("from_node", "from_port", "to_node", "to_port"):
        values.append(_require(raw.get(key), str, f"{where}.{key}"))
    return Edge(*values)


def parse(raw: Any) -> GraphDoc:
    """Dict -> GraphDoc. Structural problems raise; meaning is the validator's job."""
    _require(raw, dict, "graph")
    fmt = raw.get("format")
    if fmt != FORMAT:
        raise GraphFormatError(f"graph format {fmt!r} is not supported (this version reads format {FORMAT})")
    nodes = []
    for i, item in enumerate(_require(raw.get("nodes", []), list, "nodes")):
        where = f"nodes[{i}]"
        _require(item, dict, where)
        params = item.get("params", {})
        layout = item.get("layout", {})
        nodes.append(NodeSpec(
            uid=_require(item.get("uid"), str, f"{where}.uid"),
            class_name=_require(item.get("class_name"), str, f"{where}.class_name"),
            params=dict(_require(params, dict, f"{where}.params")),
            title=_require(item.get("title", ""), str, f"{where}.title"),
            layout=dict(_require(layout, dict, f"{where}.layout")),
        ))
    edges = tuple(_edge(e, f"edges[{i}]") for i, e in enumerate(_require(raw.get("edges", []), list, "edges")))
    event_edges = tuple(
        _edge(e, f"event_edges[{i}]") for i, e in enumerate(_require(raw.get("event_edges", []), list, "event_edges")))
    settings = dict(_require(raw.get("settings", {}), dict, "settings"))
    views = tuple(_require(raw.get("views", []), list, "views"))
    extras = {k: v for k, v in raw.items() if k not in _KNOWN_KEYS}
    return GraphDoc(tuple(nodes), edges, event_edges, settings, views, extras)
