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
from ..domain.entities.run import Run
from ..domain.value_objects import GraphStatus, RunStatus
from .ports.config_inspector import StartOption
from .ports.dataset_library import (
    DatasetInfo,
    DatasetItem,
    DatasetStats,
    DatasetSummary,
    TrainingSetInfo,
)
from .ports.dataset_tasks import DatasetTask
from .ports.graph_runtime import GraphIssue
from .ports.graph_library import SavedGraph


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
    """Outcome of the startup sweep.

    ``cleaned`` counts rows moved out of unfinished; ``adopted`` counts
    trainers that were still alive and got re-attached to instead of
    killed (docs 07 F-11).
    """

    cleaned: int
    adopted: int = 0


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
    whatever task is currently running. ``preview_path`` is the
    resolved card image (``DatasetPreviews`` port), null when the
    dataset has no usable preview."""

    info: DatasetInfo
    stats: DatasetStats | None
    sets: tuple[TrainingSetInfo, ...]
    active_tasks: tuple[DatasetTask, ...]
    preview_path: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetListResult:
    datasets: tuple[DatasetSummary, ...]
    count: int


@dataclass(frozen=True, slots=True)
class DatasetItemsResult:
    items: tuple[DatasetItem, ...]
    count: int
    limit: int | None = None   # None = every row was requested
    offset: int = 0


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
    tasks: tuple[DatasetTask, ...]
    count: int


@dataclass(frozen=True, slots=True)
class TeacherTaskParams:
    """Sampling + prompt options for ``generate_teacher`` (M8e).

    Ported from the legacy ``/tasks/start`` ``type="teacher"`` body
    (``server/routes_datasets.py``), one field per legacy parameter so
    the stored task ``params`` record stays a flat, faithful launch
    payload. ``seed``/``latent_size``/``model_type`` are shared with
    image import in the wire schema but live here for this kind, since
    they are what the builder's teacher path consumes.

    Validation (modes, ranges, prompt content) is *not* here: this is
    a DTO. ``application.teacher_prompts`` owns it, shared between the
    start use case (fail fast, before the row exists) and the task
    worker (re-assemble the prompt configs it hands to the builder).
    """

    prompt_mode: str = "list"  # 'list' | 'keywords'
    prompts: str = ""  # newline-separated, mode=list
    keywords: str = ""  # newline-separated, mode=keywords
    keywords_file: str = ""  # optional server-side word list (.txt/.csv)
    template: str = ""  # "{keywords}" placeholder template
    min_keywords: int = 3
    max_keywords: int = 10
    neg_mode: str = "list"  # 'list' | 'keywords'
    negative_prompt: str = ""  # one string for every sample, mode=list
    neg_keywords: str = ""
    neg_keywords_file: str = ""
    neg_template: str = ""
    neg_min_keywords: int = 3
    neg_max_keywords: int = 10
    cfg_min: float = 3.0
    cfg_max: float = 9.0
    steps_min: int = 20
    steps_max: int = 30
    t_mode: str = "uniform"  # uniform | low | mid | high | logit
    t_low: int = 20
    t_high: int = 999
    batch_size: int = 1
    seed: int = 42
    n_conditions: int = 10
    n_samples_per_cond: int = 1
    latent_size: int = 64
    model_type: str = "eps"  # eps | vpred


@dataclass(frozen=True, slots=True)
class StartDatasetTaskCommand:
    """Launch a dataset task. ``model`` is a checkpoint path
    relative to the resolved checkpoints dir (validated + sandboxed by
    the use case); ``image_dir`` is an absolute server-side source
    directory (validated for existence, not sandboxed -- raw images
    legitimately live outside the workspace, as in the legacy API).

    Two kinds (``application.ports.dataset_tasks.TASK_KINDS``):
    ``ingest_lora`` VAE-encodes an image directory (flat fields below);
    ``generate_teacher`` samples new trajectories from a checkpoint
    (``teacher`` carries its options, absent for the import kind)."""

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
    teacher: TeacherTaskParams | None = None


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
