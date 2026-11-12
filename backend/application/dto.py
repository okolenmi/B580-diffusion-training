"""DTOs -- the data shapes use cases accept and return.

Use cases never hand domain entities to presentation: entities carry
behaviour and an event buffer that must not leak across the boundary.
The mapping functions here are the single, symmetric conversion point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..domain.entities.graph_execution import GraphExecution
from ..domain.graph import NodeResult
from ..domain.value_objects import GraphStatus
from .ports.graph_runtime import GraphIssue
from .ports.graph_library import SavedGraph
from .ports.template_library import SavedTemplate

# --------------------------------------------------------------------------
# Use-case inputs / outputs

@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Outcome of a startup sweep of unfinished rows.

    Used by the graph-execution reconciler; ``adopted`` exists because ``adopted`` exists on a type a reconciler that adopts nothing can
    still produce: a graph run lives in its own session and can outlive
    the server that started it, so that sweep has a second outcome
    besides "row moved out of an unfinished state" and counting them
    together would hide how much work was thrown away.

    ``cleaned`` is the count that failed; ``adopted`` is the count that
    was re-attached to and is still running.

    ``still_running`` is the count that was left alone *because* something
    of it is alive and cannot be adopted -- a child whose event file is
    gone, or two children claiming one id. It is neither of the other two:
    the row did not leave an unfinished state (``cleaned``) and nobody is
    watching the process (``adopted``). Counting it as either would report
    discarded work or supervised work, and both would be false, which is
    why it is reported rather than folded in. It is 0 for a reconciler
    whose runs are threads, which cannot outlive the server.
    """

    cleaned: int
    adopted: int = 0
    still_running: int = 0


@dataclass(frozen=True, slots=True)
class RawConfig:
    """A config file's exact text (raw editor round-trip)."""

    content: str

# --------------------------------------------------------------------------
# Graphs (M4)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GraphExecutionDTO:
    """Full projection of one execution: identity, lifecycle, per-node
    results, and the graph snapshot of what actually ran."""

    execution_id: int
    status: GraphStatus
    error: str | None
    name: str | None
    results: tuple[NodeResult, ...]
    graph: dict  # {"format": 1, "nodes": [...], "edges": [...]}
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True, slots=True)
class GraphExecutionSummaryDTO:
    """List-page projection: no results, no graph snapshot."""

    execution_id: int
    status: GraphStatus
    error: str | None
    name: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def to_execution_dto(execution: GraphExecution) -> GraphExecutionDTO:
    """Map a domain entity to its full read-side projection."""
    if execution.id is None:
        raise ValueError("cannot project an unpersisted execution (no id yet)")
    return GraphExecutionDTO(
        execution_id=execution.id,
        status=execution.status,
        error=execution.error,
        name=execution.name,
        results=execution.results,
        graph=execution.graph.as_dict(),
        created_at=execution.created_at,
        updated_at=execution.updated_at,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


def to_execution_summary_dto(execution: GraphExecution) -> GraphExecutionSummaryDTO:
    if execution.id is None:
        raise ValueError("cannot project an unpersisted execution (no id yet)")
    return GraphExecutionSummaryDTO(
        execution_id=execution.id,
        status=execution.status,
        error=execution.error,
        name=execution.name,
        created_at=execution.created_at,
        updated_at=execution.updated_at,
        started_at=execution.started_at,
        finished_at=execution.finished_at,
    )


@dataclass(frozen=True, slots=True)
class ExecutionListResult:
    executions: tuple[GraphExecutionSummaryDTO, ...]
    count: int


@dataclass(frozen=True, slots=True)
class GraphMemoryPreview:
    """What admission would decide for a graph that has not started.

    Every optional field is an explicit unknown rather than a zero
    (task rule 2): a null ``peak_mb`` means this configuration has never
    been measured, which is a different fact from "it needs nothing",
    and a null ``device_demand_mb`` means there was no ledger to measure
    free space against.
    """

    fingerprint_key: str | None
    #: The remembered high-water mark for this fingerprint, allocator MB.
    peak_mb: float | None
    #: Allocator MB the run would need, and where that number came from
    #: ("stated" / "observed" / "unknown").
    demand_mb: float | None
    demand_source: str
    exploratory: bool
    #: Allocator demand plus the per-process overhead -- the quantity the
    #: ledger actually compares against free space. None with no ledger.
    device_demand_mb: float | None
    #: "fits" / "does_not_fit" / "unknown" / "exploratory".
    verdict: str
    #: Why the verdict is what it is, or None when it fits.
    reason: str | None
    capacity_mb: float | None
    free_mb: float | None
    held_mb: float | None
    foreign_reserve_mb: float | None
    #: Who holds the card, straight from the ledger snapshot the UI
    #: already reads for /health.
    holders: dict[str, dict]


@dataclass(frozen=True, slots=True)
class GraphValidationResult:
    """``ok`` is "no error-severity issue"; warnings never block."""

    ok: bool
    issues: tuple[GraphIssue, ...]


@dataclass(frozen=True, slots=True)
class DeleteGraphExecutionsResult:
    deleted: int


@dataclass(frozen=True, slots=True)
class SavedGraphDTO:
    """One library row on the wire (``graph`` verbatim as stored)."""

    name: str
    description: str
    graph: dict
    node_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SavedGraphSummaryDTO:
    """List-page projection (no payload)."""

    name: str
    description: str
    node_count: int
    created_at: datetime
    updated_at: datetime


def to_saved_graph_dto(saved: SavedGraph) -> SavedGraphDTO:
    """Map a ``SavedGraph`` port row to its projection."""
    return SavedGraphDTO(
        name=saved.name,
        description=saved.description,
        graph=dict(saved.graph),
        node_count=saved.node_count,
        created_at=saved.created_at,
        updated_at=saved.updated_at,
    )


def to_saved_graph_summary(saved) -> SavedGraphSummaryDTO:
    return SavedGraphSummaryDTO(
        name=saved.name,
        description=saved.description,
        node_count=saved.node_count,
        created_at=saved.created_at,
        updated_at=saved.updated_at,
    )


@dataclass(frozen=True, slots=True)
class SavedGraphListResult:
    graphs: tuple[SavedGraphSummaryDTO, ...]
    count: int


@dataclass(frozen=True, slots=True)
class SaveGraphResult:
    graph: SavedGraphDTO
    created: bool  # True on first save (201), False on replace (200)


@dataclass(frozen=True, slots=True)
class DeleteGraphResult:
    deleted: bool


# --------------------------------------------------------------------------
# Templates (saved pages / monitor windows)

@dataclass(frozen=True, slots=True)
class SavedTemplateDTO:
    """One template row on the wire (``doc`` verbatim as stored)."""

    name: str
    description: str
    kind: str
    doc: dict
    block_count: int
    node_count: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class SavedTemplateSummaryDTO:
    """List projection (no document)."""

    name: str
    description: str
    kind: str
    block_count: int
    node_count: int
    created_at: datetime
    updated_at: datetime


def to_saved_template_dto(saved: SavedTemplate) -> SavedTemplateDTO:
    return SavedTemplateDTO(
        name=saved.name,
        description=saved.description,
        kind=saved.kind,
        doc=dict(saved.doc),
        block_count=saved.block_count,
        node_count=saved.node_count,
        created_at=saved.created_at,
        updated_at=saved.updated_at,
    )


def to_saved_template_summary(saved: SavedTemplate) -> SavedTemplateSummaryDTO:
    return SavedTemplateSummaryDTO(
        name=saved.name,
        description=saved.description,
        kind=saved.kind,
        block_count=saved.block_count,
        node_count=saved.node_count,
        created_at=saved.created_at,
        updated_at=saved.updated_at,
    )


@dataclass(frozen=True, slots=True)
class SavedTemplateListResult:
    templates: tuple[SavedTemplateSummaryDTO, ...]
    count: int


@dataclass(frozen=True, slots=True)
class SaveTemplateResult:
    template: SavedTemplateDTO
    created: bool  # True on first save (201), False on replace (200)


@dataclass(frozen=True, slots=True)
class DeleteTemplateResult:
    deleted: bool
