"""DescribeGraphModules / ResolveTemplate -- what a page can bind to, and
whether a saved page still binds to it.

Both answer from the *real* node classes through the catalog, never from a
second description:

* ``DescribeGraphModules`` takes a graph and lists, per node instance, the
  **fields** it could expose (its settable input ports, with their current
  values and whether an edge already feeds them) and the **displays** it can
  show (its declared live views, each with the stream it reports to), plus
  the graph's own modules (memory settings, run status, memory chart). It
  is the template editor's "what is available" tray, and the monitor
  composer's "what can I watch".

* ``ResolveTemplate`` takes a *draft* document (the editor's unsaved state --
  hence a body, not a name) and says, block by block, whether its target
  still exists. A block whose node was deleted, whose class vanished, or
  whose port was renamed is reported ``missing_*`` and **kept**: the page
  still loads, the block shows as orphaned and can be removed or repointed.
  A field the graph now wires from upstream is ``wired`` (shown read-only,
  because its value no longer comes from the page).

Neither runs anything; both are cheap and safe to call on every edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import TemplateInvalidError
from ..ports.graph_catalog import GraphCatalog, NodeInfo, PortInfo
from ..ports.graph_runtime import GraphRuntime, issue_to_dict
from ...domain.graph import GraphDefinition
from ...domain.modules import (
    GRAPH_DISPLAY_BY_NAME,
    GRAPH_DISPLAYS,
    GRAPH_FIELD_BY_NAME,
    GRAPH_FIELDS,
    node_stream_id,
    wired_ports,
)
from ...domain.template import validate_template_doc
from .save_graph import normalize_graph_payload

#: Port types a person can type a value for. Everything else (a model, a
#: dataset handle, a callback) only exists as a wire and has no control.
_SETTABLE_TYPES = frozenset({"str", "int", "float", "bool", "Path"})


def is_settable(port: PortInfo) -> bool:
    """Does this input have a control a form can draw?"""
    return bool(
        port.type in _SETTABLE_TYPES or port.choices is not None or port.path_kind
    )


def _field_dict(port: PortInfo, params: dict, wired: bool) -> dict[str, Any]:
    return {
        "name": port.name,
        "type": port.type,
        "required": port.required,
        "doc": port.doc,
        "default": port.default,
        "default_repr": port.default_repr,
        "choices": list(port.choices) if port.choices is not None else None,
        "path_kind": port.path_kind,
        "visible_when": list(port.visible_when) if port.visible_when is not None else None,
        "widget_only": port.widget_only,
        "label": port.label,
        "group": port.group,
        "advanced": port.advanced,
        "unit": port.unit,
        "minimum": port.minimum,
        "maximum": port.maximum,
        "step": port.step,
        "wired": wired,
        "has_value": port.name in params,
        "value": params.get(port.name),
    }


@dataclass(frozen=True, slots=True)
class GraphModules:
    """Everything a page could bind to, for one graph."""

    nodes: tuple[dict, ...]
    graph_fields: tuple[dict, ...]
    graph_displays: tuple[dict, ...]


class DescribeGraphModules:
    def __init__(self, catalog: GraphCatalog) -> None:
        self._catalog = catalog

    def execute(self, graph: GraphDefinition) -> GraphModules:
        by_class: dict[str, NodeInfo] = {
            n.class_name: n for n in self._catalog.snapshot().nodes
        }
        nodes: list[dict] = []
        for spec in graph.nodes:
            info = by_class.get(spec.class_name)
            if info is None:
                # Unknown class: kept in the answer so a block pointing at
                # it can say "unknown class" rather than "no such node".
                nodes.append({
                    "id": spec.id, "class_name": spec.class_name,
                    "display_name": spec.class_name, "known": False,
                    "fields": [], "displays": [], "stream": node_stream_id(spec.id),
                })
                continue
            inputs = self._catalog.resolved_inputs(spec.class_name, spec.params) or info.inputs
            wired = wired_ports(graph.edges, spec.id)
            fields = [
                _field_dict(port, spec.params, port.name in wired)
                for port in inputs
                if is_settable(port)
            ]
            stream = self._stream_for(graph, spec.id, by_class)
            nodes.append({
                "id": spec.id,
                "class_name": spec.class_name,
                "display_name": info.display_name,
                "known": True,
                "fields": fields,
                "displays": [dict(m, stream=stream) for m in info.modules],
                "stream": stream,
            })
        memory = graph.memory.as_dict()
        graph_fields = tuple(
            dict(f, value=memory.get(f["name"].split(".", 1)[1]))
            for f in GRAPH_FIELDS
        )
        return GraphModules(
            nodes=tuple(nodes),
            graph_fields=graph_fields,
            graph_displays=tuple(dict(d) for d in GRAPH_DISPLAYS),
        )

    @staticmethod
    def _stream_for(graph: GraphDefinition, node_id: str, by_class: dict[str, NodeInfo]) -> str:
        """The stream this node's displays arrive on.

        The runtime hands a node with declared modules a handle onto
        ``node:<id>``. But a graph that still wires a MonitorNode into the
        node's ``monitor`` input sends its reports to that monitor's id
        instead (the old route, kept working), so a page must follow the
        wire to find them."""
        for edge in graph.edges:
            if edge.to_node == node_id and edge.to_port == "monitor":
                source = graph.node(edge.from_node)
                info = by_class.get(source.class_name) if source else None
                if source is not None and info is not None and any(
                    p.name == "monitor_id" for p in info.inputs
                ):
                    monitor_id = source.params.get("monitor_id")
                    if isinstance(monitor_id, str) and monitor_id:
                        return monitor_id
        return node_stream_id(node_id)


@dataclass(frozen=True, slots=True)
class TemplateResolution:
    modules: GraphModules
    #: One entry per block, in order: ``{"id", "status", "detail"}``.
    blocks: tuple[dict, ...]
    #: Graph validation issues (errors + warnings) -- "would this run".
    issues: tuple[dict, ...]


#: Block statuses. ``ok`` and ``wired`` still render; the ``missing_*``
#: ones render as an orphan placeholder naming what is gone.
STATUS_OK = "ok"
STATUS_WIRED = "wired"
STATUS_MISSING_NODE = "missing_node"
STATUS_UNKNOWN_CLASS = "unknown_class"
STATUS_MISSING_MODULE = "missing_module"


class ResolveTemplate:
    def __init__(self, *, describe: DescribeGraphModules, runtime: GraphRuntime) -> None:
        self._describe = describe
        self._runtime = runtime

    def execute(self, doc: dict, *, graph: dict | None = None) -> TemplateResolution:
        problems = validate_template_doc(doc)
        if problems:
            raise TemplateInvalidError(
                f"template has {len(problems)} problem(s)", details=problems
            )
        source = graph if graph is not None else doc.get("graph")
        if source is None:
            definition = GraphDefinition()
        else:
            try:
                definition = GraphDefinition.from_dict(normalize_graph_payload(source))
            except ValueError as exc:  # a bad memory block
                raise TemplateInvalidError(f"graph: {exc}", details=[str(exc)]) from exc
        modules = self._describe.execute(definition)
        nodes = {n["id"]: n for n in modules.nodes}

        blocks = tuple(
            self._status(block, nodes) for block in doc.get("blocks", [])
        )
        issues: tuple[dict, ...] = ()
        if doc.get("kind", "template") == "template" and definition.nodes:
            issues = tuple(issue_to_dict(i) for i in self._runtime.validate(definition))
        return TemplateResolution(modules=modules, blocks=blocks, issues=issues)

    @staticmethod
    def _status(block: dict, nodes: dict[str, dict]) -> dict:
        ref = block["ref"]
        out = {"id": block["id"], "status": STATUS_OK, "detail": ""}
        kind, name = ref["kind"], ref["name"]
        if ref["scope"] == "graph":
            known = GRAPH_FIELD_BY_NAME if kind == "field" else GRAPH_DISPLAY_BY_NAME
            if name not in known:
                out.update(status=STATUS_MISSING_MODULE,
                           detail=f"the graph has no {kind} named {name!r}")
            return out
        node = nodes.get(ref["node"])
        if node is None:
            out.update(status=STATUS_MISSING_NODE,
                       detail=f"node {ref['node']!r} is not in the graph")
            return out
        if not node["known"]:
            out.update(status=STATUS_UNKNOWN_CLASS,
                       detail=f"node class {node['class_name']!r} is not installed")
            return out
        if kind == "field":
            field = next((f for f in node["fields"] if f["name"] == name), None)
            if field is None:
                out.update(status=STATUS_MISSING_MODULE,
                           detail=f"{node['display_name']} has no field {name!r}")
            elif field["wired"]:
                out.update(status=STATUS_WIRED,
                           detail=f"{name!r} is fed by another node; its value is not set here")
        elif not any(d["name"] == name for d in node["displays"]):
            out.update(status=STATUS_MISSING_MODULE,
                       detail=f"{node['display_name']} has no live view {name!r}")
        return out
