"""GraphDefinition -- the submitted node graph, as data: the graph document of rework/contracts/graph-json.md.

The document itself (nodes with a ``uid``, data and event edges, settings, views, addons) is defined and parsed
by ``next/graph/document.py``; this module wraps it with what the server needs: the memory settings read from the
graph's ``memory.*`` and ``device.default`` settings, and a decoder for the snapshots of older executions.
No validation happens here: ``GraphRuntime.validate()`` gives one complete issue list instead of an exception on the
first problem.

The import below is the one place the domain names the new node tree (see ``infrastructure/graph/tree.py``); it is pure
Python and pulls in no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from next.graph.document import FORMAT as GRAPH_FORMAT
from next.graph.document import Edge, GraphDoc, GraphFormatError, NodeSpec, parse

from .memory_settings import MemorySettings

__all__ = ["GRAPH_FORMAT", "Edge", "GraphDefinition", "GraphFormatError", "NodeResult", "NodeSpec"]

LEGACY_KEY = "legacy_snapshot"


@dataclass(frozen=True, slots=True)
class NodeResult:
    """Outcome of one node's build(), as reported back to the caller.

    ``node_id`` is the node's ``uid``. ``outputs`` is already JSON-described (tensors and handles become
    ``{"_type", "_repr"}`` summaries -- the real objects only ever live inside the executing process).
    ``duration_ms`` is wall time of that node's build call.
    """

    node_id: str
    ok: bool
    outputs: dict = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class GraphDefinition:
    """A whole submitted graph, immutable: the document plus what the server derives from it."""

    doc: GraphDoc = field(default_factory=GraphDoc)

    @property
    def nodes(self) -> tuple[NodeSpec, ...]:
        return self.doc.nodes

    @property
    def edges(self) -> tuple[Edge, ...]:
        return self.doc.edges

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self.doc.uids)

    @property
    def memory(self) -> MemorySettings:
        """The graph is the configurable object (ADR 0005): its budget is in its own settings."""
        return MemorySettings.from_settings(self.doc.settings)

    @property
    def is_legacy(self) -> bool:
        """A snapshot of an earlier format, kept for the history only: it has no nodes to compare results with."""
        return LEGACY_KEY in self.doc.extras

    def node(self, uid: str) -> NodeSpec | None:
        return self.doc.node(uid)

    def edges_to(self, uid: str) -> tuple[Edge, ...]:
        return tuple(self.doc.edges_to(uid))

    def with_params(self, params_by_uid: dict[str, dict]) -> GraphDefinition:
        """The same graph with these nodes' params replaced (a uid not named keeps its own)."""
        nodes = tuple(replace(n, params=params_by_uid.get(n.uid, n.params)) for n in self.doc.nodes)
        return GraphDefinition(replace(self.doc, nodes=nodes))

    def with_settings(self, settings: dict) -> GraphDefinition:
        return GraphDefinition(replace(self.doc, settings=dict(settings)))

    def as_dict(self) -> dict:
        legacy = self.doc.extras.get(LEGACY_KEY)
        return dict(legacy) if legacy is not None else self.doc.as_dict()

    @classmethod
    def from_dict(cls, payload: dict) -> GraphDefinition:
        """A submitted or saved document. Raises ``GraphFormatError`` when it is not one this version reads."""
        return cls(parse(payload))

    @classmethod
    def from_stored(cls, payload: dict) -> GraphDefinition:
        """The snapshot kept with an execution. One written by an earlier format cannot be run again, but the history
        must still list it: it comes back without nodes, and ``as_dict`` returns it as it was stored."""
        try:
            return cls.from_dict(payload)
        except GraphFormatError:
            return cls(GraphDoc(extras={LEGACY_KEY: dict(payload)}))
