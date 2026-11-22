"""Graph runs, validation and the saved-graph library."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...application.ports.graph_runtime import GraphIssue
from ...domain.graph import GRAPH_FORMAT, GraphDefinition
from .memory import MemoryOverridesIn


class GraphRunIn(BaseModel):
    """Submission body for ``/validate`` and ``/run``: the graph document (docs/contracts/graph-json.md), whose
    ``settings`` carry the memory budget, and this run's own overrides of it."""

    graph: dict[str, Any]
    memory_overrides: MemoryOverridesIn | None = None
    #: What a start does when the card has no room: ``parallel`` refuses, ``auto`` plans the run, ``queue`` plans it
    #: whenever anything else lives or waits (``/validate`` ignores it).
    mode: Literal["auto", "parallel", "queue"] = "auto"
    #: What the run is called in the list (blank: none).
    name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _graph_reads_and_override_stays_in_range(self) -> GraphRunIn:
        """The graph must be a document this version reads, and the *merged* memory request must be ordered, not just
        each half: an override that lifts the floor above the graph's ceiling would otherwise reach the use case and
        die there as a 500 (domain errors escaping a use case are always a bug), instead of here as a 422 naming
        both numbers."""
        memory = self.to_definition().memory
        if self.memory_overrides is None:
            return self
        minimum = (
            self.memory_overrides.vram_min_mb
            if self.memory_overrides.vram_min_mb is not None
            else memory.vram_min_mb
        )
        maximum = (
            self.memory_overrides.vram_max_mb
            if self.memory_overrides.vram_max_mb is not None
            else memory.vram_max_mb
        )
        if not isinstance(maximum, str) and minimum > maximum:
            raise ValueError(
                f"vram_min_mb ({minimum}) exceeds vram_max_mb ({maximum}) "
                "with the overrides applied"
            )
        return self

    def to_definition(self) -> GraphDefinition:
        return GraphDefinition.from_dict(self.graph)


class DiagnosticsIn(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsOut(BaseModel):
    messages: dict[str, list[str]]


class GraphIssueOut(BaseModel):
    severity: str
    code: str
    message: str
    node_id: str | None = None
    edge_index: int | None = None
    param: str | None = None
    plane: str = "data"


class ValidateOut(BaseModel):
    """Always 200: ``ok=false`` means the run endpoint would reject
    this submission (422 with these same issues)."""

    ok: bool
    issues: list[GraphIssueOut]


class LibraryGraphIn(BaseModel):
    """``PUT /library/{name}``: a graph payload plus an optional
    description. Unknown top-level keys ride along verbatim (the
    library stores what was submitted; validation happens at run)."""

    model_config = ConfigDict(extra="allow")

    format: int | None = None
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    description: str = ""

    def to_payload(self) -> dict:
        payload = dict(self.model_extra or {})
        payload["format"] = self.format if self.format is not None else GRAPH_FORMAT
        payload["nodes"] = list(self.nodes)
        payload["edges"] = list(self.edges)
        return payload


class SavedGraphOut(BaseModel):
    name: str
    description: str = ""
    graph: dict[str, Any]
    node_count: int
    created_at: datetime
    updated_at: datetime


class SavedGraphSummaryOut(BaseModel):
    name: str
    description: str = ""
    node_count: int
    created_at: datetime
    updated_at: datetime


class SavedGraphListOut(BaseModel):
    graphs: list[SavedGraphSummaryOut]
    count: int


class DeleteGraphOut(BaseModel):
    deleted: bool


def graph_issue_out(issue: GraphIssue) -> GraphIssueOut:
    return GraphIssueOut(
        severity=issue.severity,
        code=issue.code,
        message=issue.message,
        node_id=issue.node_id,
        edge_index=issue.edge_index,
        param=issue.param,
        plane=issue.plane,
    )


def saved_graph_out(dto) -> SavedGraphOut:
    """``dto``: application ``SavedGraphDTO``."""
    return SavedGraphOut(
        name=dto.name,
        description=dto.description,
        graph=dict(dto.graph),
        node_count=dto.node_count,
        created_at=dto.created_at,
        updated_at=dto.updated_at,
    )


def saved_graph_summary_out(dto) -> SavedGraphSummaryOut:
    return SavedGraphSummaryOut(
        name=dto.name,
        description=dto.description,
        node_count=dto.node_count,
        created_at=dto.created_at,
        updated_at=dto.updated_at,
    )


def saved_graph_list_out(result) -> SavedGraphListOut:
    """``result``: application ``SavedGraphListResult``."""
    return SavedGraphListOut(
        graphs=[saved_graph_summary_out(g) for g in result.graphs],
        count=result.count,
    )


__all__ = ["GraphRunIn", "DiagnosticsIn", "DiagnosticsOut", "GraphIssueOut", "ValidateOut", "LibraryGraphIn", "SavedGraphOut", "SavedGraphSummaryOut", "SavedGraphListOut", "DeleteGraphOut", "graph_issue_out", "saved_graph_out", "saved_graph_summary_out", "saved_graph_list_out"]
