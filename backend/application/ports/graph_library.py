"""GraphLibrary port -- named server-side graph storage.

Replaces the legacy editor's browser-only persistence (localStorage
``ng_graph_v1``): graphs survive browsers, are shareable, and carry a
stamped format version.

``graph`` is stored **verbatim** -- the submitted ``{"format": 1,
"nodes": [...], "edges": [...]}`` payload with unknown keys preserved.
Saving never validates class names or edges: a graph saved while a
class is missing still loads later, because validation runs at *run*
time (forward compatibility, and the run endpoint is the one place that
must be strict).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

MAX_GRAPH_NAME = 120
"""Longest accepted library name (validated by ``normalize_graph_name``)."""


def normalize_graph_name(raw: str) -> str:
    """Trimmed, length-bounded library name.

    A ValueError here means "the use case turns it into
    ``invalid_query``" -- the rule lives next to the port so save/get/
    delete agree on what a name is.
    """
    name = raw.strip()
    if not name:
        raise ValueError("graph name must not be empty")
    if len(name) > MAX_GRAPH_NAME:
        raise ValueError(
            f"graph name must be at most {MAX_GRAPH_NAME} characters, got {len(name)}"
        )
    return name


@dataclass(frozen=True, slots=True)
class SavedGraph:
    """One library row (``graph`` is the decoded stored payload)."""

    name: str
    description: str
    graph: dict
    created_at: datetime
    updated_at: datetime

    @property
    def node_count(self) -> int:
        """Nodes in the stored graph (0 when absent/malformed payload)."""
        nodes = self.graph.get("nodes")
        return len(nodes) if isinstance(nodes, list) else 0


class GraphLibrary(ABC):
    """Row store for saved graphs; repositories do not raise."""

    @abstractmethod
    def save(
        self, name: str, graph: dict, *, description: str = ""
    ) -> tuple[SavedGraph, bool]:
        """Upsert; returns ``(row, created)`` -- ``created`` distinguishes
        201 from 200 at the API boundary."""
        raise NotImplementedError

    @abstractmethod
    def get(self, name: str) -> SavedGraph | None:
        raise NotImplementedError

    @abstractmethod
    def list(self) -> tuple[SavedGraph, ...]:
        """Every saved graph, most recently updated first."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Remove one graph; ``False`` when it did not exist."""
        raise NotImplementedError
