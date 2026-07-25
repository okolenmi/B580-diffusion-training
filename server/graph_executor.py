"""GraphExecutor: runs a graph the editor submits.

Unlike nodegraph_introspect.py, this module is NOT side-effect-free by
design -- executing a node graph means constructing real models, loading
real checkpoints, and (for SupervisedLoRATrainerNode) running real
training steps. That's the point of "Run"; it's meant to have effects.
What it does NOT do is touch core/, manager/, or server's production
config/training path -- it only calls .build() on nodes/ classes, same
boundary the rest of nodes/ already keeps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import nodegraph_registry


class GraphError(Exception):
    """Graph-shape problem (unknown class, missing edge target, cycle) --
    caught before any node's build() runs, so nothing partially executes."""


@dataclass
class NodeSpec:
    id: str
    class_name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeSpec:
    from_node: str
    from_port: str
    to_node: str
    to_port: str


@dataclass
class NodeResult:
    node_id: str
    ok: bool
    outputs: dict[str, Any] = field(default_factory=dict)   # real Python objects, this process only
    error: str | None = None


def _topological_order(node_ids: list[str], edges: list[EdgeSpec]) -> list[str]:
    depends_on: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for e in edges:
        depends_on[e.to_node].add(e.from_node)

    ordered: list[str] = []
    remaining = set(node_ids)
    while remaining:
        ready = [nid for nid in remaining if depends_on[nid] <= set(ordered)]
        if not ready:
            raise GraphError(f"Cycle detected among nodes: {sorted(remaining)}")
        ready.sort()  # stable order when multiple nodes are ready at once
        ordered.extend(ready)
        remaining -= set(ready)
    return ordered


def _is_compatible(from_cls: type, from_port: str, to_cls: type, to_port: str) -> None:
    """Raises GraphError if the real declared types don't line up. Real
    issubclass() against the actual Port.type objects -- not string
    comparison -- so e.g. a FusedOptimizerHandle output correctly satisfies
    an OptimizerHandle input, being an actual subclass of it."""
    if from_port not in from_cls.OUTPUTS:
        raise GraphError(f"{from_cls.__name__} has no output port '{from_port}'")
    if to_port not in to_cls.INPUTS:
        raise GraphError(f"{to_cls.__name__} has no input port '{to_port}'")
    out_type = from_cls.OUTPUTS[from_port].type
    in_type = to_cls.INPUTS[to_port].type
    if in_type is Any or out_type is Any:
        return
    if isinstance(out_type, type) and isinstance(in_type, type) and issubclass(out_type, in_type):
        return
    raise GraphError(
        f"{from_cls.__name__}.{from_port} ({getattr(out_type, '__name__', out_type)}) "
        f"is not compatible with {to_cls.__name__}.{to_port} ({getattr(in_type, '__name__', in_type)})"
    )


def _describe(value: Any) -> Any:
    """JSON-safe view of a build() output for the response payload. Plain
    JSON types pass through; anything else (tensors, Handle objects, model
    wrappers) becomes a short type+repr description -- the real object
    only ever lives inside this process, feeding the next node's build()
    call, never serialized."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_describe(v) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): _describe(v) for k, v in list(value.items())[:20]}
    return {"_type": type(value).__name__, "_repr": repr(value)[:200]}


class GraphExecutor:

    def __init__(self, nodes: list[NodeSpec], edges: list[EdgeSpec]):
        self.nodes = {n.id: n for n in nodes}
        self.edges = edges
        self._registry = nodegraph_registry.get_registry()

    def _resolve_class(self, spec: NodeSpec) -> type:
        cls = self._registry.get(spec.class_name)
        if cls is None:
            raise GraphError(f"Unknown node class '{spec.class_name}' (not in the registry)")
        return cls

    def run(self) -> list[NodeResult]:
        node_ids = list(self.nodes.keys())
        for e in self.edges:
            if e.from_node not in self.nodes or e.to_node not in self.nodes:
                raise GraphError(f"Edge references a node not in this graph: {e}")
            _is_compatible(self._resolve_class(self.nodes[e.from_node]), e.from_port,
                            self._resolve_class(self.nodes[e.to_node]), e.to_port)

        order = _topological_order(node_ids, self.edges)
        outputs_by_node: dict[str, dict[str, Any]] = {}
        results: list[NodeResult] = []

        for node_id in order:
            spec = self.nodes[node_id]
            cls = self._resolve_class(spec)
            inputs = dict(spec.params)
            for e in self.edges:
                if e.to_node == node_id:
                    inputs[e.to_port] = outputs_by_node[e.from_node][e.from_port]

            try:
                outputs = cls().build(**inputs)
            except Exception as exc:  # noqa: BLE001 -- a failing node is a normal outcome to report, not a crash
                results.append(NodeResult(node_id=node_id, ok=False,
                                           error=f"{type(exc).__name__}: {exc}"))
                return results  # downstream nodes depend on this one's outputs; stop here
            outputs_by_node[node_id] = outputs
            results.append(NodeResult(node_id=node_id, ok=True,
                                       outputs={k: _describe(v) for k, v in outputs.items()}))

        return results
