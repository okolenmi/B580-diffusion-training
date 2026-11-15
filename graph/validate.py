"""Graph validation: one deterministic issue list over real node classes."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .addons import ADDON_ID, GRAPH, SLOT_PREFIX, end_of, slot_sockets, slots_of, split_port
from .document import UID_PATTERN, Edge, GraphDoc
from .node import Environment, Node, Resolution
from .ports import type_name, types_compatible
from .registry import NodeRegistry
from .settings import SettingsRegistry

ERROR = "error"
WARNING = "warning"

_JSON_TYPES = (bool, int, float, str, list, dict, tuple)


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str
    uid: str | None = None
    port: str | None = None
    edge_index: int | None = None
    plane: str = "data"

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message, "uid": self.uid,
                "port": self.port, "edge_index": self.edge_index, "plane": self.plane}


@dataclass
class Validation:
    issues: list[Issue] = field(default_factory=list)
    resolutions: dict[str, Resolution] = field(default_factory=dict)
    order: list[str] | None = None

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors


def topological_order(uids: list[str], edges: list[Edge], first: frozenset[str] = frozenset()) -> list[str]:
    """Kahn's algorithm, ties broken by uid, with the nodes in ``first`` ahead of the others that are ready at the same
    time. Raises ValueError naming the stuck nodes."""
    known = set(uids)
    depends: dict[str, set[str]] = {u: set() for u in uids}
    for e in edges:
        if e.from_node in known and e.to_node in known:
            depends[e.to_node].add(e.from_node)
    ordered: list[str] = []
    placed: set[str] = set()
    remaining = set(uids)
    while remaining:
        ready = sorted((u for u in remaining if depends[u] <= placed), key=lambda u: (u not in first, u))
        if not ready:
            raise ValueError(f"cycle among nodes: {sorted(remaining)}")
        ordered.extend(ready)
        placed.update(ready)
        remaining.difference_update(ready)
    return ordered


def literal_mismatch(port_type: Any, value: Any) -> bool:
    """True when a JSON literal cannot satisfy a primitive port type. None always passes."""
    if value is None or port_type is Any or not isinstance(port_type, type):
        return False
    if port_type not in _JSON_TYPES and port_type is not Path:
        return False
    if port_type is float:
        return isinstance(value, bool) or not isinstance(value, (int, float))
    if port_type is int:
        return isinstance(value, bool) or not isinstance(value, int)
    if port_type is bool:
        return not isinstance(value, bool)
    if port_type is str:
        return not isinstance(value, str)
    if port_type is Path:
        return not isinstance(value, (str, Path))
    if port_type is tuple:
        return not isinstance(value, (list, tuple))
    return not isinstance(value, port_type)


def validate(doc: GraphDoc, registry: NodeRegistry, settings: SettingsRegistry | None = None,
             env: Environment | None = None) -> Validation:
    return _Validator(doc, registry, settings or registry.settings_registry(), env).run()


class _Validator:
    """Seven passes over a graph document, each one adding issues and leaving what the next needs: the nodes' own
    shapes (pass 1), their addons (2), their params (3), the data wires (4), the event wires (5), the build order (6), the
    settings (7)."""

    def __init__(self, doc: GraphDoc, registry: NodeRegistry, settings: SettingsRegistry, env: Environment | None) -> None:
        self.doc, self.settings, self.env = doc, settings, env
        self.classes, _ = registry.load()
        self.addon_classes = registry.addons()
        self.addon_ports: dict[str, dict] = {}
        self.result = Validation()
        self.cls_of: dict[str, type[Node]] = {}
        self.shaped: dict[str, Resolution] = {}
        self.unique_uids = 0

    def run(self) -> Validation:
        self._shapes()
        self._addons()
        self._params()
        self._data_wires()
        self._event_wires()
        self._order()
        self._settings()
        return self.result

    def add(self, severity: str, code: str, message: str, **where: Any) -> None:
        self.result.issues.append(Issue(severity, code, message, **where))

    def _shapes(self) -> None:
        seen: set[str] = set()
        for spec in self.doc.nodes:
            if not UID_PATTERN.match(spec.uid or ""):
                self.add(ERROR, "invalid_uid", f"node uid {spec.uid!r} is not of the form n_xxxxxx", uid=spec.uid)
            if spec.uid in seen:
                self.add(ERROR, "duplicate_uid", f"node uid {spec.uid!r} appears more than once", uid=spec.uid)
            seen.add(spec.uid)
            cls = self.classes.get(spec.class_name)
            if cls is None:
                self.add(ERROR, "unknown_class", f"no node class named {spec.class_name!r} is registered", uid=spec.uid)
                continue
            self.cls_of[spec.uid] = cls
            try:
                res = cls.resolve(dict(spec.params), self.env)
            except Exception as exc:  # noqa: BLE001 -- the node's bug becomes an issue
                self.add(ERROR, "shape_resolution_failed",
                         f"{spec.class_name}.resolve raised {type(exc).__name__}: {exc}", uid=spec.uid)
                continue
            if res.state == "ready":
                res = dataclasses.replace(res, inputs={**res.inputs, **slot_sockets(cls)})
            self.result.resolutions[spec.uid] = res
            if res.state == "needs_choice":
                self.add(ERROR, "needs_choice", res.message or f"{spec.class_name} needs a choice", uid=spec.uid, port=res.focus)
            elif res.state == "error":
                self.add(ERROR, "resolve_error", res.message or f"{spec.class_name} could not be resolved", uid=spec.uid,
                         port=res.focus)
            else:
                self.shaped[spec.uid] = res
        self.unique_uids = len(seen)

    def _params(self) -> None:
        for spec in self.doc.nodes:
            res = self.shaped.get(spec.uid)
            if res is None:
                continue
            fed = {e.to_port for e in self.doc.edges_to(spec.uid)}
            for name, port in res.inputs.items():
                provided = name in spec.params
                if port.required and not provided and name not in fed:
                    self.add(ERROR, "missing_required_input",
                             f"{spec.class_name}.{name} is required but neither a param nor a wire provides it",
                             uid=spec.uid, port=name)
                if provided:
                    self._literal(spec.uid, spec.class_name, name, port, spec.params[name])
            for key in spec.params:
                if key not in res.inputs:
                    self.add(ERROR, "unknown_param", f"{spec.class_name} has no input named {key!r}", uid=spec.uid, port=key)

    def _addons(self) -> None:
        hosts = [(s.uid, s.class_name, self.cls_of.get(s.uid), s.addons) for s in self.doc.nodes if s.uid in self.cls_of]
        for uid, label, cls, specs in [*hosts, (GRAPH, GRAPH, None, self.doc.addons)]:
            self._host_addons(uid, label, cls, specs)

    def _host_addons(self, uid: str, label: str, cls: Any, specs: tuple) -> None:
        slots = {s.name: s for s in slots_of(cls)}
        ports = self.addon_ports.setdefault(uid, {})
        enabled = [s.kind for s in specs if s.enabled]
        filled: dict[str, int] = {}
        fed = {e.to_port for e in self.doc.edges_to(uid)}
        ids: set[str] = set()
        for spec in specs:
            where = {"uid": uid, "port": spec.id}
            if not ADDON_ID.match(spec.id):
                self.add(ERROR, "addon_bad_id", f"addon id {spec.id!r} on {label} must be lower case letters, digits and _", **where)
            if spec.id in ids:
                self.add(ERROR, "addon_duplicate_id", f"addon id {spec.id!r} appears twice on {label}", **where)
            ids.add(spec.id)
            cls_a = self.addon_classes.get(spec.kind)
            if cls_a is None:
                self.add(ERROR, "addon_unknown_kind", f"no addon class named {spec.kind!r} is registered", **where)
                continue
            for name, port in cls_a.INPUTS.items():
                ports[f"{spec.id}.{name}"] = port
            if cls_a.SLOT not in slots:
                self.add(ERROR, "addon_slot_missing", f"{label} has no addon slot {cls_a.SLOT!r} for {spec.kind}", **where)
                continue
            if cls_a.FITS and label not in cls_a.FITS:
                self.add(ERROR, "addon_not_offered", f"{spec.kind} is not offered on {label} (it fits {list(cls_a.FITS)})", **where)
            if not spec.enabled:
                continue
            filled[cls_a.SLOT] = filled.get(cls_a.SLOT, 0) + 1
            for kind in cls_a.REQUIRES:
                if kind not in enabled:
                    self.add(ERROR, "addon_requires", f"{spec.kind} on {label} needs {kind} on the same host", **where)
            for kind in cls_a.CONFLICTS:
                if kind in enabled:
                    self.add(ERROR, "addon_conflict", f"{spec.kind} on {label} cannot be combined with {kind}", **where)
            for name, port in cls_a.INPUTS.items():
                wire = f"{spec.id}.{name}" in fed
                if name in spec.params:
                    self._literal(uid, label, f"{spec.id}.{name}", port, spec.params[name])
                elif port.required and not wire:
                    self.add(ERROR, "missing_required_input",
                             f"{spec.kind}.{name} on {label} is required but neither a param nor a wire provides it",
                             uid=uid, port=f"{spec.id}.{name}")
            for key in spec.params:
                if key not in cls_a.INPUTS:
                    self.add(ERROR, "unknown_param", f"{spec.kind} has no parameter named {key!r}", uid=uid, port=f"{spec.id}.{key}")
        for edge in self.doc.edges_to(uid):
            if edge.to_port.startswith(SLOT_PREFIX):
                name = edge.to_port[len(SLOT_PREFIX):]
                filled[name] = filled.get(name, 0) + 1
        for name, count in filled.items():
            if count > 1 and name in slots and not slots[name].many:
                self.add(ERROR, "addon_slot_taken", f"slot {name!r} of {label} holds one; it is filled {count} times", uid=uid, port=name)

    def _input_port(self, uid: str, res: Any, name: str) -> Any:
        """A node's own input, or an addon's parameter (``a1.start``) on it."""
        port = res.inputs.get(name) if res else None
        if port is None and split_port(name):
            port = self.addon_ports.get(uid, {}).get(name)
        return port

    def _literal(self, uid: str, label: str, name: str, port: Any, value: Any) -> None:
        where = {"uid": uid, "port": name}
        label = f"{label}.{name}"
        if port.choices is not None and value is not None and value not in port.choices:
            self.add(ERROR, "invalid_choice", f"{label}={value!r} is not one of {list(port.choices)}", **where)
        if literal_mismatch(port.type, value):
            self.add(ERROR, "type_mismatch", f"{label} expects {type_name(port.type)}, got {type(value).__name__}", **where)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if port.minimum is not None and value < port.minimum:
                self.add(ERROR, "out_of_range", f"{label}={value} is below {port.minimum}", **where)
            if port.maximum is not None and value > port.maximum:
                self.add(ERROR, "out_of_range", f"{label}={value} is above {port.maximum}", **where)

    def _data_wires(self) -> None:
        wires_in: dict[tuple[str, str], int] = {}
        for index, edge in enumerate(self.doc.edges):
            src, dst = self.doc.node(edge.from_node), self.doc.node(edge.to_node)
            if src is None or dst is None:
                missing = edge.from_node if src is None else edge.to_node
                self.add(ERROR, "edge_unknown_node", f"wire references node {missing!r}, which is not in the graph",
                         uid=missing, edge_index=index)
                continue
            out_res, in_res = self.shaped.get(src.uid), self.shaped.get(dst.uid)
            out_port = out_res.outputs.get(edge.from_port) if out_res else None
            in_port = self._input_port(dst.uid, in_res, edge.to_port)
            if out_res and out_port is None:
                self.add(ERROR, "unknown_output_port", f"{src.class_name} has no output port {edge.from_port!r}",
                         uid=src.uid, port=edge.from_port, edge_index=index)
            if in_res and in_port is None:
                self.add(ERROR, "unknown_input_port", f"{dst.class_name} has no input port {edge.to_port!r}",
                         uid=dst.uid, port=edge.to_port, edge_index=index)
            if out_port and in_port and not types_compatible(out_port.type, in_port.type):
                self.add(ERROR, "incompatible_types",
                         f"{src.class_name}.{edge.from_port} ({type_name(out_port.type)}) does not satisfy "
                         f"{dst.class_name}.{edge.to_port} ({type_name(in_port.type)})",
                         uid=dst.uid, port=edge.to_port, edge_index=index)
            if in_port is not None and not in_port.multi:
                key = (dst.uid, edge.to_port)
                wires_in[key] = wires_in.get(key, 0) + 1
                if wires_in[key] == 2:
                    self.add(ERROR, "multiple_wires_to_input",
                             f"{dst.class_name}.{edge.to_port} takes one wire; it has several", uid=dst.uid,
                             port=edge.to_port, edge_index=index)

    def _event_wires(self) -> None:
        for index, edge in enumerate(self.doc.event_edges):
            (src_ok, src_cls, src_label), (dst_ok, dst_cls, dst_label) = (
                end_of(self.doc, address, self.classes, self.addon_classes) for address in (edge.from_node, edge.to_node))
            if not (src_ok and dst_ok):
                missing = edge.from_node if not src_ok else edge.to_node
                self.add(ERROR, "edge_unknown_node", f"event wire references {missing!r}, which is not in the graph",
                         uid=missing, edge_index=index, plane="event")
                continue
            out_ev = src_cls.EVENTS_OUT.get(edge.from_port) if src_cls else None
            in_ev = dst_cls.EVENTS_IN.get(edge.to_port) if dst_cls else None
            if src_cls and out_ev is None:
                self.add(ERROR, "unknown_event_out", f"{src_label} has no event output {edge.from_port!r}",
                         uid=edge.from_node, port=edge.from_port, edge_index=index, plane="event")
            if dst_cls and in_ev is None:
                self.add(ERROR, "unknown_event_in", f"{dst_label} has no event input {edge.to_port!r}",
                         uid=edge.to_node, port=edge.to_port, edge_index=index, plane="event")
            if out_ev and in_ev and not issubclass(out_ev.payload, in_ev.payload):
                self.add(ERROR, "event_incompatible",
                         f"{src_label}.{edge.from_port} ({out_ev.payload.__name__}) does not satisfy "
                         f"{dst_label}.{edge.to_port} ({in_ev.payload.__name__})",
                         uid=edge.to_node, port=edge.to_port, edge_index=index, plane="event")
        if self.doc.event_edges and _has_cycle([(e.from_node, e.to_node) for e in self.doc.event_edges]):
            self.add(WARNING, "event_cycle", "the event wires form a cycle; the runtime stops a re-entrant loop", plane="event")

    def _order(self) -> None:
        """The build order: data wires decide, and events only break ties; skipped while uids repeat."""
        doc = self.doc
        if self.unique_uids != len(doc.nodes):
            return
        first = frozenset(u for u, cls in self.cls_of.items() if any(ev.scoped for ev in cls.EVENTS_IN.values()))
        try:
            data_order = topological_order(doc.uids, list(doc.edges), first)
        except ValueError as exc:
            self.add(ERROR, "cycle", str(exc))
            return
        self.result.order = data_order
        if not doc.event_edges:
            return
        # Prefer to build an event's subscriber before its emitter, so a handler is in place when the emitter first
        # emits; data edges always win, so fall back when they disagree.
        host = lambda address: address.partition(".")[0]  # noqa: E731 -- an addon is built with its host
        reverse = [Edge(host(e.to_node), e.to_port, host(e.from_node), e.from_port) for e in doc.event_edges
                   if host(e.to_node) != host(e.from_node)]
        try:
            self.result.order = topological_order(doc.uids, list(doc.edges) + reverse, first)
        except ValueError:
            self.add(WARNING, "event_order",
                     "event wires cannot all be satisfied by build order; an event emitted before its "
                     "subscriber is built fails the run", plane="event")

    def _settings(self) -> None:
        for severity, key, message in self.settings.validate(self.doc.settings):
            self.add(severity, "invalid_setting" if severity == ERROR else "unknown_setting", f"{key}: {message}", port=key)
        for message in self.settings.errors:
            self.add(WARNING, "settings_conflict", message)


def _has_cycle(pairs: list[tuple[str, str]]) -> bool:
    graph: dict[str, list[str]] = {}
    for a, b in pairs:
        graph.setdefault(a, []).append(b)
    state: dict[str, int] = {}

    def visit(node: str) -> bool:
        state[node] = 1
        for nxt in graph.get(node, ()):
            if state.get(nxt) == 1 or (state.get(nxt) is None and visit(nxt)):
                return True
        state[node] = 2
        return False

    return any(state.get(n) is None and visit(n) for n in list(graph))
