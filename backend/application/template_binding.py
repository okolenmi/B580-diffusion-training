"""Applying a page's values to its graph -- the rule that makes a template safe.

A template page exposes some fields and hides the rest. Running it submits
*values for the exposed fields only*: a request that sets a field the page
does not show is refused, not applied. That is the whole point of a
template for someone who should not have to meet the graph -- the graph's
other knobs hold the values the template's author saved, and a stale tab or
a hand-built request cannot reach past the page.

Pure functions over domain objects (no I/O, no catalog): ``exposed_fields``
reads the document's blocks, ``apply_values`` returns a new
``GraphDefinition``.
"""

from __future__ import annotations


from ..domain.graph import GraphDefinition, GraphNodeSpec
from ..domain.memory_settings import MemorySettings
from ..domain.modules import GRAPH_FIELD_BY_NAME, wired_ports


def exposed_fields(doc: dict) -> tuple[set[tuple[str, str]], set[str]]:
    """``({(node_id, port), ...}, {memory_key, ...})`` for the field blocks
    a document places. ``memory_key`` is the setting without its
    ``memory.`` prefix (``vram_max_mb``)."""
    node_fields: set[tuple[str, str]] = set()
    graph_fields: set[str] = set()
    for block in doc.get("blocks", []):
        ref = block["ref"]
        if ref["kind"] != "field":
            continue
        if ref["scope"] == "node":
            node_fields.add((ref["node"], ref["name"]))
        elif ref["name"] in GRAPH_FIELD_BY_NAME:
            graph_fields.add(ref["name"].split(".", 1)[1])
    return node_fields, graph_fields


def _apply_node_values(
    graph: GraphDefinition,
    node_id: str,
    ports: object,
    allowed_nodes: set[tuple[str, str]],
    problems: list[str],
) -> dict | None:
    """One node's port values -> its new params, appending why not.

    `None` means "no answer for this node" (unknown id, or ports not an
    object) -- the caller then omits it from the new-params map, which is
    what the inlined loop's `continue` did: the node keeps its own
    `spec.params` downstream via `new_params.get(n.id, n.params)`, and
    since `problems` is non-empty the whole graph is returned unchanged
    anyway.

    Split out of `apply_values` for C901 (its own complexity budget),
    not for behaviour: the checks, their order, and the exact problem
    strings are unchanged, so a caller still sees the same list in the
    same sequence.
    """
    spec = graph.node(node_id)
    if spec is None:
        problems.append(f"values.nodes.{node_id}: the graph has no such node")
        return None
    if not isinstance(ports, dict):
        problems.append(f"values.nodes.{node_id}: must be an object of port values")
        return None

    wired = wired_ports(graph.edges, node_id)
    params = dict(spec.params)
    for port, value in ports.items():
        if (node_id, port) not in allowed_nodes:
            problems.append(
                f"values.nodes.{node_id}.{port}: this page does not expose that field"
            )
        elif port in wired:
            problems.append(
                f"values.nodes.{node_id}.{port}: the field is fed by another node"
            )
        elif value is None:
            params.pop(port, None)
        else:
            params[port] = value
    return params


def apply_values(
    graph: GraphDefinition, doc: dict, values: dict | None
) -> tuple[GraphDefinition, list[str]]:
    """The graph with ``values`` applied, and every reason it could not be.

    ``values`` is ``{"nodes": {node_id: {port: value}}, "memory": {key:
    value}}``. A ``None`` value clears the param (the port's default
    applies). Problems are collected, not raised one at a time; when the
    list is non-empty the returned graph is the *unchanged* input.
    """
    values = values or {}
    problems: list[str] = []
    allowed_nodes, allowed_memory = exposed_fields(doc)

    extra = set(values) - {"nodes", "memory"}
    if extra:
        problems.append(f"values: unknown section(s) {sorted(extra)}")

    node_values = values.get("nodes") or {}
    memory_values = values.get("memory") or {}
    if not isinstance(node_values, dict) or not isinstance(memory_values, dict):
        return graph, ["values: 'nodes' and 'memory' must be objects"]

    new_params: dict[str, dict] = {}
    for node_id, ports in node_values.items():
        applied = _apply_node_values(graph, node_id, ports, allowed_nodes, problems)
        if applied is not None:
            new_params[node_id] = applied

    memory = graph.memory
    if memory_values:
        for key in memory_values:
            if key not in allowed_memory:
                problems.append(f"values.memory.{key}: this page does not expose that setting")
        if not problems:
            merged = {**memory.as_dict(), **{k: v for k, v in memory_values.items() if v is not None}}
            try:
                memory = MemorySettings.from_dict(merged)
            except ValueError as exc:
                problems.append(f"values.memory: {exc}")

    if problems:
        return graph, problems
    nodes = tuple(
        GraphNodeSpec(id=n.id, class_name=n.class_name, params=new_params.get(n.id, n.params))
        for n in graph.nodes
    )
    return GraphDefinition(nodes=nodes, edges=graph.edges, memory=memory), []
