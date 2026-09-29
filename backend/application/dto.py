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
    cache_done: int | None
    cache_total: int | None
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
        cache_done=run.cache_done,
        cache_total=run.cache_total,
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


# Valid values for StartTrainingCommand.start_from (validated by the
# use case -- one source of truth, not duplicated into pydantic).
START_FROM_OPTIONS: tuple[str, ...] = (
    "teacher",
    "student",
    "resume",
    "lora_checkpoint",
)


@dataclass(frozen=True, slots=True)
class StartTrainingCommand:
    """Launch request; relative ``config_path`` anchors at project root."""

    config_path: str
    start_from: str = "teacher"
    reset_optimizer: bool = False


@dataclass(frozen=True, slots=True)
class LogResult:
    """Tail of a run's log; empty string when the file does not exist."""

    log: str


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """Outcome of the startup sweep: rows moved out of unfinished."""

    cleaned: int
