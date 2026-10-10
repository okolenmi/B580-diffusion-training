"""TemplateLibrary port -- named server-side template storage.

Same posture as ``GraphLibrary`` and for the same reasons: the document
(``doc``) is stored **verbatim** apart from the format stamp the use case
adds, unknown keys survive a round trip, and the repository never raises
for a missing row (``get`` answers None, ``delete`` answers False).

A template carries a snapshot of its graph inside ``doc["graph"]`` rather
than a foreign key into the graph library: deleting or editing a library
graph must not break a page someone built on it, and "sync this page from
its graph" is an explicit action (``source_graph`` remembers which).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from ..limits import MAX_TEMPLATE_NAME

__all__ = ["MAX_TEMPLATE_NAME", "SavedTemplate", "TemplateLibrary", "normalize_template_name"]


def normalize_template_name(raw: str) -> str:
    """Trimmed, length-bounded template name (ValueError -> the use case
    turns it into ``invalid_query``)."""
    name = raw.strip()
    if not name:
        raise ValueError("template name must not be empty")
    if len(name) > MAX_TEMPLATE_NAME:
        raise ValueError(
            f"template name must be at most {MAX_TEMPLATE_NAME} characters, got {len(name)}"
        )
    return name


@dataclass(frozen=True, slots=True)
class SavedTemplate:
    """One library row (``doc`` is the decoded stored document)."""

    name: str
    description: str
    kind: str  # "template" | "monitor" -- a column so a list can filter without decoding
    doc: dict
    created_at: datetime
    updated_at: datetime

    @property
    def block_count(self) -> int:
        blocks = self.doc.get("blocks")
        return len(blocks) if isinstance(blocks, list) else 0

    @property
    def node_count(self) -> int:
        graph = self.doc.get("graph")
        nodes = graph.get("nodes") if isinstance(graph, dict) else None
        return len(nodes) if isinstance(nodes, list) else 0


class TemplateLibrary(ABC):
    @abstractmethod
    def save(
        self, name: str, doc: dict, *, kind: str, description: str = ""
    ) -> tuple[SavedTemplate, bool]:
        """Upsert; ``(row, created)``."""
        raise NotImplementedError

    @abstractmethod
    def get(self, name: str) -> SavedTemplate | None:
        raise NotImplementedError

    @abstractmethod
    def list_templates(self, kind: str | None = None) -> tuple[SavedTemplate, ...]:
        """Most recently updated first; ``kind`` filters."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> bool:
        raise NotImplementedError
