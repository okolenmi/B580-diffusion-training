"""GraphDefinition -- the submitted node graph, as data.

The shape an editor submits (and an execution stores): nodes with their
class + widget params, and the edges wiring output ports to input
ports. Deliberately dumb data with no validation in ``__post_init__``:
*all* checking goes through ``GraphRuntime.validate()`` so a caller gets
one complete issue list instead of an exception on the first problem
(see ``docs/design/backend/05-graph-runtime.md`` section 4).

This module is pure Python -- no ``nodes.*`` imports (the class registry
lives behind an infrastructure adapter) and nothing JSON/pydantic-specific:
``as_dict``/``from_dict`` are the symmetric storage/wire conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory_settings import MemorySettings

GRAPH_FORMAT = 2
"""Version stamped into stored graph payloads (``{"format": 2, ...}``).

Format 1 graphs load with default ``MemorySettings`` (the graph has no
memory settings, so the server uses the observed peak or refuses).
Format 2 adds the ``memory`` key to the payload.
"""


@dataclass(frozen=True, slots=True)
class GraphNodeSpec:
    """One node placement: stable ``id`` within the graph, the stable
    ``class_name`` it resolves against (``__name__`` of a real Node
    subclass -- see docs/design/10-node-surface-and-precision-control.md
    section 11.5), and raw JSON widget values."""

    id: str
    class_name: str
    params: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdgeSpec:
    """A wire from one node's output port to another's input port."""

    from_node: str
    from_port: str
    to_node: str
    to_port: str


@dataclass(frozen=True, slots=True)
class NodeResult:
    """Outcome of one node's build(), as reported back to the caller.

    ``outputs`` is already JSON-described (tensors/handles become
    ``{"_type", "_repr"}`` summaries -- the real objects only ever live
    inside the executing process, feeding the next node's build()).
    ``duration_ms`` is wall time of that node's build call.
    """

    node_id: str
    ok: bool
    outputs: dict = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class GraphDefinition:
    """A whole submitted graph: nodes + edges + this graph's own memory
    settings, immutable."""

    nodes: tuple[GraphNodeSpec, ...] = ()
    edges: tuple[GraphEdgeSpec, ...] = ()
    #: The graph is the configurable object (ADR 0005): each graph holds
    #: its own budget here, not in a global setting.
    memory: MemorySettings = field(default_factory=MemorySettings)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    def node(self, node_id: str) -> GraphNodeSpec | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def edges_to(self, node_id: str) -> tuple[GraphEdgeSpec, ...]:
        """Edges whose target is ``node_id`` (the executor's wiring)."""
        return tuple(e for e in self.edges if e.to_node == node_id)

    def as_dict(self) -> dict:
        """Storage/wire shape: ``{"format": 2, "nodes": [...],
        "edges": [...], "memory": {...}}`` (params dicts passed through
        verbatim)."""
        return {
            "format": GRAPH_FORMAT,
            "nodes": [
                {"id": n.id, "class_name": n.class_name, "params": dict(n.params)}
                for n in self.nodes
            ],
            "edges": [
                {
                    "from_node": e.from_node,
                    "from_port": e.from_port,
                    "to_node": e.to_node,
                    "to_port": e.to_port,
                }
                for e in self.edges
            ],
            "memory": self.memory.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> GraphDefinition:
        """Parse a stored/submitted payload back into specs.

        Tolerant by design (it reads our own snapshots): missing
        ``params`` defaults to ``{}``, unknown keys are ignored -- the
        authoritative shape check is ``validate()``, run before any
        execution, never this decoder.

        Format 1 graphs (no ``memory`` key) load with default
        ``MemorySettings``. Format 2 graphs carry their own settings.
        """
        nodes = tuple(
            GraphNodeSpec(
                id=str(raw.get("id", "")),
                class_name=str(raw.get("class_name", "")),
                params=dict(raw.get("params") or {}),
            )
            for raw in payload.get("nodes") or []
        )
        edges = tuple(
            GraphEdgeSpec(
                from_node=str(raw.get("from_node", "")),
                from_port=str(raw.get("from_port", "")),
                to_node=str(raw.get("to_node", "")),
                to_port=str(raw.get("to_port", "")),
            )
            for raw in payload.get("edges") or []
        )
        return cls(
            nodes=nodes,
            edges=edges,
            memory=MemorySettings.from_dict(payload.get("memory")),
        )
