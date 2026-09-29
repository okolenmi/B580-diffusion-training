"""DTOs -- the data shapes use cases accept and return.

Use cases never hand domain entities to presentation: entities carry
behaviour and an event buffer that must not leak across the boundary.
The mapping functions here are the single, symmetric conversion point.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..domain.entities.run import Run
from ..domain.value_objects import RunStatus


@dataclass(frozen=True, slots=True)
class RunDTO:
    """Immutable projection of a run for read-side consumers."""

    id: int
    status: RunStatus
    config_path: str
    mode: str
    phase: str | None
    total_steps: int
    done_steps: int
    current_loss: float | None
    avg_loss: float | None
    pid: int | None
    exit_code: int | None
    error: str | None
    log_path: str | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


def to_run_dto(run: Run) -> RunDTO:
    """Map a domain entity to its read-side projection."""
    if run.id is None:
        raise ValueError("cannot project an unpersisted run (no id yet)")
    return RunDTO(
        id=run.id,
        status=run.status,
        config_path=run.config_path,
        mode=run.mode,
        phase=run.phase,
        total_steps=run.total_steps,
        done_steps=run.done_steps,
        current_loss=run.current_loss,
        avg_loss=run.avg_loss,
        pid=run.pid,
        exit_code=run.exit_code,
        error=run.error,
        log_path=run.log_path,
        created_at=run.created_at,
        updated_at=run.updated_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


# --------------------------------------------------------------------------
# Use-case inputs / outputs
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ListRunsQuery:
    """Read-side query; ``status`` stays a string until the use case
    validates it -- query strings are untrusted input."""

    limit: int = 50
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ListRunsResult:
    runs: tuple[RunDTO, ...]
    count: int


@dataclass(frozen=True, slots=True)
class DeleteRunsResult:
    deleted: int
