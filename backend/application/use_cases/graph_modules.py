"""DescribeGraphModules / ResolveTemplate -- what a page can bind to, and
whether a saved page still binds to it.

Both answer from the node tree's own catalog and resolve, never from a
second description:

* ``DescribeGraphModules`` takes a graph and lists, per node instance, the
  **fields** it could expose (its settable input ports, with their current
  values and whether an edge already feeds them) and the **displays** it can
  show (its declared live views, each on the stream ``node:<uid>``), plus
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
from ..ports.graph_runtime import GraphRuntime, issue_to_dict
from ..ports.node_catalog import NodeCatalog
from ...domain.graph import GraphDefinition
from ...domain.modules import (
    GRAPH_DISPLAY_BY_NAME,
    GRAPH_DISPLAYS,
    GRAPH_FIELD_BY_NAME,
    GRAPH_FIELDS,
    node_stream_id,
)
from ...domain.template import validate_template_doc
from .save_graph import normalize_graph_payload


def _field_dict(port: dict, params: dict, wired: bool) -> dict[str, Any]:
    """A catalog port as the template editor's field: the port's own description plus this node's current state."""
    name = port["name"]
    return {
        **{key: port.get(key) for key in (
            "name", "type", "required", "doc", "default", "choices", "picker", "visible_when", "widget_only",
            "label", "group", "advanced", "unit", "minimum", "maximum", "step")},
        "wired": wired,
        "has_value": name in params,
        "value": params.get(name),
    }


@dataclass(frozen=True, slots=True)
class GraphModules:
    """Everything a page could bind to, for one graph."""

    nodes: tuple[dict, ...]
    graph_fields: tuple[dict, ...]
    graph_displays: tuple[dict, ...]


class DescribeGraphModules:
    def __init__(self, catalog: NodeCatalog) -> None:
        self._catalog = catalog

    def execute(self, graph: GraphDefinition) -> GraphModules:
        by_class = {n["class_name"]: n for n in self._catalog.catalog()["nodes"]}
        nodes: list[dict] = []
        for spec in graph.nodes:
            stream = node_stream_id(spec.uid)
            info = by_class.get(spec.class_name)
            if info is None:
                # Unknown class: kept in the answer so a block pointing at
                # it can say "unknown class" rather than "no such node".
                nodes.append({
                    "id": spec.uid, "class_name": spec.class_name,
                    "display_name": spec.class_name, "known": False,
                    "fields": [], "displays": [], "stream": stream,
                })
                continue
            wired = {edge.to_port for edge in graph.edges_to(spec.uid)}
            nodes.append({
                "id": spec.uid,
                "class_name": spec.class_name,
                "display_name": info["display_name"],
                "known": True,
                "fields": [
                    _field_dict(port, spec.params, port["name"] in wired)
                    for port in self._inputs(info, spec, graph) if port.get("field")
                ],
                "displays": [dict(module, stream=stream) for module in info["modules"]],
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

    def _inputs(self, info: dict, spec: Any, graph: GraphDefinition) -> list[dict]:
        """The node's inputs as these params shape them. A dynamic node resolves; when it cannot, the static ports
        stand, as in the editor (which keeps the last good shape)."""
        if info.get("dynamic") is None:
            return info["inputs"]
        resolved = self._catalog.resolve(spec.class_name, spec.params, graph.doc.settings)
        return info["inputs"] if resolved["state"] == "error" else resolved["inputs"]


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
            except ValueError as exc:  # not a document this version reads, or a bad memory setting
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
