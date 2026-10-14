"""Graph validation: one deterministic issue list over real node classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .document import UID_PATTERN, Edge, GraphDoc
from .node import Environment, Node, Resolution
from .ports import Port, type_name, types_compatible
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


def topological_order(uids: list[str], edges: list[Edge]) -> list[str]:
    """Kahn's algorithm, ties broken by uid. Raises ValueError naming the stuck nodes."""
    known = set(uids)
    depends: dict[str, set[str]] = {u: set() for u in uids}
    for e in edges:
        if e.from_node in known and e.to_node in known:
            depends[e.to_node].add(e.from_node)
    ordered: list[str] = []
    placed: set[str] = set()
    remaining = set(uids)
    while remaining:
        ready = sorted(u for u in remaining if depends[u] <= placed)
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
    result = Validation()
    issues = result.issues
    classes, _ = registry.load()
    settings = settings or registry.settings_registry()

    def add(severity, code, message, **kw):
        issues.append(Issue(severity, code, message, **kw))

    # pass 1: identity and shape
    seen: set[str] = set()
    shaped: dict[str, Resolution] = {}
    cls_of: dict[str, type[Node]] = {}
    for spec in doc.nodes:
        if not UID_PATTERN.match(spec.uid or ""):
            add(ERROR, "invalid_uid", f"node uid {spec.uid!r} is not of the form n_xxxxxx", uid=spec.uid)
        if spec.uid in seen:
            add(ERROR, "duplicate_uid", f"node uid {spec.uid!r} appears more than once", uid=spec.uid)
        seen.add(spec.uid)
        cls = classes.get(spec.class_name)
        if cls is None:
            add(ERROR, "unknown_class", f"no node class named {spec.class_name!r} is registered", uid=spec.uid)
            continue
        cls_of[spec.uid] = cls
        try:
            res = cls.resolve(dict(spec.params), env)
        except Exception as exc:  # noqa: BLE001 -- the node's bug becomes an issue
            add(ERROR, "shape_resolution_failed",
                f"{spec.class_name}.resolve raised {type(exc).__name__}: {exc}", uid=spec.uid)
            continue
        result.resolutions[spec.uid] = res
        if res.state == "needs_choice":
            add(ERROR, "needs_choice", res.message or f"{spec.class_name} needs a choice", uid=spec.uid, port=res.focus)
        elif res.state == "error":
            add(ERROR, "resolve_error", res.message or f"{spec.class_name} could not be resolved", uid=spec.uid,
                port=res.focus)
        else:
            shaped[spec.uid] = res

    # pass 2: params against the resolved inputs
    for spec in doc.nodes:
        res = shaped.get(spec.uid)
        if res is None:
            continue
        fed = {e.to_port for e in doc.edges_to(spec.uid)}
        for name, port in res.inputs.items():
            provided = name in spec.params
            if port.required and not provided and name not in fed:
                add(ERROR, "missing_required_input",
                    f"{spec.class_name}.{name} is required but neither a param nor a wire provides it",
                    uid=spec.uid, port=name)
            if provided:
                value = spec.params[name]
                if port.choices is not None and value is not None and value not in port.choices:
                    add(ERROR, "invalid_choice", f"{spec.class_name}.{name}={value!r} is not one of {list(port.choices)}",
                        uid=spec.uid, port=name)
                if literal_mismatch(port.type, value):
                    add(ERROR, "type_mismatch",
                        f"{spec.class_name}.{name} expects {type_name(port.type)}, got {type(value).__name__}",
                        uid=spec.uid, port=name)
                if port.minimum is not None and isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and value < port.minimum:
                    add(ERROR, "out_of_range", f"{spec.class_name}.{name}={value} is below {port.minimum}",
                        uid=spec.uid, port=name)
                if port.maximum is not None and isinstance(value, (int, float)) and not isinstance(value, bool) \
                        and value > port.maximum:
                    add(ERROR, "out_of_range", f"{spec.class_name}.{name}={value} is above {port.maximum}",
                        uid=spec.uid, port=name)
        for key in spec.params:
            if key not in res.inputs:
                add(ERROR, "unknown_param", f"{spec.class_name} has no input named {key!r}", uid=spec.uid, port=key)

    # pass 3: data edges
    per_input: dict[tuple[str, str], int] = {}
    for index, edge in enumerate(doc.edges):
        src, dst = doc.node(edge.from_node), doc.node(edge.to_node)
        if src is None or dst is None:
            missing = edge.from_node if src is None else edge.to_node
            add(ERROR, "edge_unknown_node", f"wire references node {missing!r}, which is not in the graph",
                uid=missing, edge_index=index)
            continue
        out_res, in_res = shaped.get(src.uid), shaped.get(dst.uid)
        out_port = out_res.outputs.get(edge.from_port) if out_res else None
        in_port = in_res.inputs.get(edge.to_port) if in_res else None
        if out_res and out_port is None:
            add(ERROR, "unknown_output_port", f"{src.class_name} has no output port {edge.from_port!r}",
                uid=src.uid, port=edge.from_port, edge_index=index)
        if in_res and in_port is None:
            add(ERROR, "unknown_input_port", f"{dst.class_name} has no input port {edge.to_port!r}",
                uid=dst.uid, port=edge.to_port, edge_index=index)
        if out_port and in_port and not types_compatible(out_port.type, in_port.type):
            add(ERROR, "incompatible_types",
                f"{src.class_name}.{edge.from_port} ({type_name(out_port.type)}) does not satisfy "
                f"{dst.class_name}.{edge.to_port} ({type_name(in_port.type)})",
                uid=dst.uid, port=edge.to_port, edge_index=index)
        if in_port is not None and not in_port.multi:
            key = (dst.uid, edge.to_port)
            per_input[key] = per_input.get(key, 0) + 1
            if per_input[key] == 2:
                add(ERROR, "multiple_wires_to_input",
                    f"{dst.class_name}.{edge.to_port} takes one wire; it has several", uid=dst.uid,
                    port=edge.to_port, edge_index=index)

    # pass 4: event edges
    for index, edge in enumerate(doc.event_edges):
        src, dst = doc.node(edge.from_node), doc.node(edge.to_node)
        if src is None or dst is None:
            missing = edge.from_node if src is None else edge.to_node
            add(ERROR, "edge_unknown_node", f"event wire references node {missing!r}, which is not in the graph",
                uid=missing, edge_index=index, plane="event")
            continue
        src_cls, dst_cls = cls_of.get(src.uid), cls_of.get(dst.uid)
        out_ev = src_cls.EVENTS_OUT.get(edge.from_port) if src_cls else None
        in_ev = dst_cls.EVENTS_IN.get(edge.to_port) if dst_cls else None
        if src_cls and out_ev is None:
            add(ERROR, "unknown_event_out", f"{src.class_name} has no event output {edge.from_port!r}",
                uid=src.uid, port=edge.from_port, edge_index=index, plane="event")
        if dst_cls and in_ev is None:
            add(ERROR, "unknown_event_in", f"{dst.class_name} has no event input {edge.to_port!r}",
                uid=dst.uid, port=edge.to_port, edge_index=index, plane="event")
        if out_ev and in_ev and not issubclass(out_ev.payload, in_ev.payload):
            add(ERROR, "event_incompatible",
                f"{src.class_name}.{edge.from_port} ({out_ev.payload.__name__}) does not satisfy "
                f"{dst.class_name}.{edge.to_port} ({in_ev.payload.__name__})",
                uid=dst.uid, port=edge.to_port, edge_index=index, plane="event")
    if doc.event_edges and _has_cycle([(e.from_node, e.to_node) for e in doc.event_edges]):
        add(WARNING, "event_cycle", "the event wires form a cycle; the runtime stops a re-entrant loop", plane="event")

    # pass 5: acyclicity of the data plane
    if len(seen) == len(doc.nodes):
        try:
            data_order = topological_order(doc.uids, list(doc.edges))
        except ValueError as exc:
            add(ERROR, "cycle", str(exc))
        else:
            result.order = data_order
            if doc.event_edges:
                # Prefer to build an event's subscriber before its emitter, so a handler is in place
                # when the emitter first emits; data edges always win, so fall back when they disagree.
                reverse = [Edge(e.to_node, e.to_port, e.from_node, e.from_port) for e in doc.event_edges]
                try:
                    result.order = topological_order(doc.uids, list(doc.edges) + reverse)
                except ValueError:
                    add(WARNING, "event_order",
                        "event wires cannot all be satisfied by build order; an event emitted before its "
                        "subscriber is built fails the run", plane="event")

    # pass 6: settings
    for severity, key, message in settings.validate(doc.settings):
        add(severity, "invalid_setting" if severity == ERROR else "unknown_setting", f"{key}: {message}", port=key)
    for message in settings.errors:
        add(WARNING, "settings_conflict", message)
    return result


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
