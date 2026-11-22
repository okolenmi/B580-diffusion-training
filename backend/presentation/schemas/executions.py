"""Graph executions: summaries, results, commands, streams."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ...application.dto import GraphExecutionDTO, GraphExecutionSummaryDTO
from ...domain.value_objects import GraphStatus


class NodeResultOut(BaseModel):
    node_id: str
    ok: bool
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


class ExecutionSummaryOut(BaseModel):
    execution_id: int
    status: GraphStatus
    error: str | None = None
    name: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    #: ``{app, version, values}`` when an app started the run, None for the editor's Run.
    origin: dict[str, Any] | None = None


class ExecutionOut(ExecutionSummaryOut):
    results: list[NodeResultOut]
    graph: dict[str, Any]
    memory: dict[str, Any] | None = None


class ActionIn(BaseModel):
    """Which node's action to trigger. Nothing else travels: the node's declaration builds the event."""

    node: str = Field(min_length=1, max_length=64)
    action: str = Field(min_length=1, max_length=64)


class ActionQueuedOut(BaseModel):
    command_id: str
    node: str
    action: str
    #: Always "queued": whether the node handled it arrives on the node's stream as ``{"kind": "action", "id", "status"}`` frames.
    status: str = "queued"


class ExecutionRenameIn(BaseModel):
    name: str | None = Field(default=None, max_length=200)


class ExecutionListOut(BaseModel):
    executions: list[ExecutionSummaryOut]
    count: int


class ExecutionStreamsOut(BaseModel):
    streams: list[str]


class ExecutionFramesOut(BaseModel):
    stream: str
    frames: list[dict[str, Any]]


class DeleteExecutionsOut(BaseModel):
    deleted: int


def execution_summary_out(dto: GraphExecutionSummaryDTO) -> ExecutionSummaryOut:
    return ExecutionSummaryOut(
        execution_id=dto.execution_id,
        status=dto.status,
        error=dto.error,
        name=dto.name,
        created_at=dto.created_at,
        updated_at=dto.updated_at,
        started_at=dto.started_at,
        finished_at=dto.finished_at,
        origin=None if dto.origin is None else dict(dto.origin),
    )


def execution_out(dto: GraphExecutionDTO) -> ExecutionOut:
    return ExecutionOut(
        **execution_summary_out(dto).model_dump(),
        results=[
            NodeResultOut(
                node_id=result.node_id,
                ok=result.ok,
                outputs=dict(result.outputs),
                error=result.error,
                duration_ms=result.duration_ms,
            )
            for result in dto.results
        ],
        graph=dict(dto.graph),
        memory=None if dto.memory is None else dict(dto.memory),
    )


def execution_list_out(result) -> ExecutionListOut:
    """``result``: application ``ExecutionListResult``."""
    return ExecutionListOut(
        executions=[execution_summary_out(e) for e in result.executions],
        count=result.count,
    )


__all__ = ["NodeResultOut", "ExecutionSummaryOut", "ExecutionOut", "ActionIn", "ActionQueuedOut", "ExecutionRenameIn", "ExecutionListOut", "ExecutionStreamsOut", "ExecutionFramesOut", "DeleteExecutionsOut", "execution_summary_out", "execution_out", "execution_list_out"]
