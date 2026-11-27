"""AppLibrary port -- named server-side app storage.

Same posture as ``GraphLibrary`` and for the same reasons: the document (``doc``) is stored **verbatim** apart from the format stamp the
use case adds, unknown keys survive a round trip, and the repository never raises for a missing row (``get`` answers None, ``delete``
answers False).

An app carries a snapshot of its graph inside ``doc["graph"]`` rather than a foreign key into the graph library: deleting or editing a
library graph must not break an app someone built on it, and "update from graph" is an explicit action (``source_graph`` remembers
which). ``version`` counts snapshots: it grows when a save changes the graph's hash (``domain/app.graph_hash``), never for layout. Using an app
(typing into its fields) is not a save: those values are kept apart (``put_values``) and a save of the document or of its graph discards them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from ...domain.board import iter_cards
from ..limits import MAX_APP_NAME

__all__ = ["MAX_APP_NAME", "AppLibrary", "SavedApp", "normalize_app_name"]


def normalize_app_name(raw: str) -> str:
    """Trimmed, length-bounded app name (ValueError -> the use case turns it into ``invalid_query``)."""
    name = raw.strip()
    if not name:
        raise ValueError("app name must not be empty")
    if len(name) > MAX_APP_NAME:
        raise ValueError(f"app name must be at most {MAX_APP_NAME} characters, got {len(name)}")
    return name


@dataclass(frozen=True, slots=True)
class SavedApp:
    """One library row (``doc`` is the decoded stored document)."""

    name: str
    description: str
    version: int
    graph_hash: str
    doc: dict
    created_at: datetime
    updated_at: datetime
    #: What was set on the page since the last save (``{"nodes": ..., "memory": ...}``); a save makes the document the new baseline and empties it.
    values: dict = field(default_factory=dict)

    @property
    def card_count(self) -> int:
        """Cards that point at something (containers and their tabs are not counted)."""
        return sum(1 for _ in iter_cards(self.doc))

    @property
    def node_count(self) -> int:
        graph = self.doc.get("graph")
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        return len(nodes) if isinstance(nodes, list) else 0


class AppLibrary(ABC):
    @abstractmethod
    def save(self, name: str, doc: dict, *, graph_hash: str, description: str = "") -> tuple[SavedApp, bool]:
        """Upsert; ``(row, created)``. A first save is version 1; a replace keeps the version unless ``graph_hash`` changed."""
        raise NotImplementedError

    @abstractmethod
    def put_values(self, name: str, values: dict) -> bool:
        """Keep the page's current values beside the document (no version, no ``updated_at``); False when there is no such app."""
        raise NotImplementedError

    @abstractmethod
    def get(self, name: str) -> SavedApp | None:
        raise NotImplementedError

    @abstractmethod
    def list_apps(self) -> tuple[SavedApp, ...]:
        """Most recently updated first."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> bool:
        raise NotImplementedError
