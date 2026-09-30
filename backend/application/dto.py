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
from .ports.config_inspector import StartOption


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


@dataclass(frozen=True, slots=True)
class RawConfig:
    """A config file's exact text (raw editor round-trip)."""

    content: str


@dataclass(frozen=True, slots=True)
class LastFinishedRun:
    """Most recent run that reached a terminal state (oldest-first
    queries never surface one: the caller scans newest-first)."""

    id: int
    config_path: str
    mode: str
    done_steps: int
    total_steps: int
    avg_loss: float | None
    status: str


@dataclass(frozen=True, slots=True)
class StartOptionsResult:
    """The "continue from" picker's data: what each option would use
    and whether it exists, plus enough run history to warn about
    unfinished work."""

    start_from: dict[str, StartOption]
    has_unfinished_run: bool
    last_finished: LastFinishedRun | None


# --------------------------------------------------------------------------
# Datasets (M3b)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetDetail:
    """One round-trip for a dataset page: identity, counts, sets, and
    whatever task is currently running."""

    info: "DatasetInfo"  # noqa: F821 -- application.ports.dataset_library
    stats: "DatasetStats | None"  # noqa: F821
    sets: tuple  # tuple[TrainingSetInfo, ...]
    active_tasks: tuple  # tuple[DatasetTask, ...]


@dataclass(frozen=True, slots=True)
class DatasetListResult:
    datasets: tuple  # tuple[DatasetSummary, ...]
    count: int


@dataclass(frozen=True, slots=True)
class DatasetItemsResult:
    items: tuple  # tuple[DatasetItem, ...]
    count: int


@dataclass(frozen=True, slots=True)
class DiscardItemsResult:
    deleted: int


@dataclass(frozen=True, slots=True)
class BulkUpdateResult:
    updated: int


@dataclass(frozen=True, slots=True)
class CommitResult:
    set_id: int
    set_name: str
    added: int


@dataclass(frozen=True, slots=True)
class DatasetTaskListResult:
    tasks: tuple  # tuple[DatasetTask, ...]
    count: int


@dataclass(frozen=True, slots=True)
class StartDatasetTaskCommand:
    """Launch an ingestion task. ``model`` is a checkpoint path
    relative to the resolved checkpoints dir (validated + sandboxed by
    the use case); ``image_dir`` is an absolute server-side source
    directory (validated for existence, not sandboxed -- raw images
    legitimately live outside the workspace, as in the legacy API)."""

    dataset: str
    kind: str = "ingest_lora"
    image_dir: str = ""
    model: str = ""
    recursive: bool = True
    resize_mode: str = "center_crop"
    latent_size: int = 64
    neg_prompt: str = ""
    model_type: str = "eps"
    seed: int = 42
    max_aspect_ratio: float = 2.0
