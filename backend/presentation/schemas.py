"""Response schemas (pydantic) -- the wire shape of this API.

One model per resource/operation, mapped explicitly from application
DTOs by the ``*_out`` helpers: the boundary is visible, and a DTO
change never silently reshapes responses.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from ..application.dto import RunDTO
from ..domain.value_objects import RunStatus


class RunOut(BaseModel):
    id: int
    status: RunStatus
    config_path: str
    mode: str
    phase: str | None = None
    total_steps: int
    done_steps: int
    current_loss: float | None = None
    avg_loss: float | None = None
    cache_done: int | None = None
    cache_total: int | None = None
    pid: int | None = None
    exit_code: int | None = None
    error: str | None = None
    log_path: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ListRunsOut(BaseModel):
    runs: list[RunOut]
    count: int


class DeleteRunsOut(BaseModel):
    deleted: int


class StartRunIn(BaseModel):
    """Launch request. ``start_from`` stays a plain string: the use
    case validates it (one source of truth for allowed values)."""

    config_path: str = Field(min_length=1)
    start_from: str = "teacher"
    reset_optimizer: bool = False


class StopRunIn(BaseModel):
    force: bool = False


class RunLogOut(BaseModel):
    log: str


class HealthOut(BaseModel):
    status: str
    version: str


def run_out(dto: RunDTO) -> RunOut:
    """Map application DTO -> response model."""
    return RunOut(
        id=dto.id,
        status=dto.status,
        config_path=dto.config_path,
        mode=dto.mode,
        phase=dto.phase,
        total_steps=dto.total_steps,
        done_steps=dto.done_steps,
        current_loss=dto.current_loss,
        avg_loss=dto.avg_loss,
        cache_done=dto.cache_done,
        cache_total=dto.cache_total,
        pid=dto.pid,
        exit_code=dto.exit_code,
        error=dto.error,
        log_path=dto.log_path,
        created_at=dto.created_at,
        updated_at=dto.updated_at,
        started_at=dto.started_at,
        finished_at=dto.finished_at,
    )
