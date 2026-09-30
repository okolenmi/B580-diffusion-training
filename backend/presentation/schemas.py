"""Response schemas (pydantic) -- the wire shape of this API.

One model per resource/operation, mapped explicitly from application
DTOs by the ``*_out`` helpers: the boundary is visible, and a DTO
change never silently reshapes responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..application.dto import (
    DatasetDetail,
    RunDTO,
    StartDatasetTaskCommand,
    StartOptionsResult,
)
from ..application.ports.asset_store import AssetCatalog, AssetBrowse
from ..application.ports.dataset_library import (
    DatasetInfo,
    DatasetItem,
    DatasetStats,
    DatasetSummary,
    TrainingSetInfo,
)
from ..application.ports.dataset_tasks import DatasetTask
from ..application.ports.settings_store import SettingsChanges, SettingsView
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


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


class ConfigPatchIn(BaseModel):
    """Partial update: ``overrides`` mirrors the config's nested
    structure and is deep-merged into the existing file."""

    path: str = Field(min_length=1)
    overrides: dict[str, Any] = Field(default_factory=dict)


class ConfigRawIn(BaseModel):
    path: str = Field(min_length=1)
    content: str


class ConfigRawOut(BaseModel):
    content: str


class ConfigSavedOut(BaseModel):
    ok: bool = True


class ConfigOptionsOut(BaseModel):
    """The field schema; values come from ``GET /config``."""

    options: list[dict[str, Any]]


class StartOptionOut(BaseModel):
    path: str
    available: bool
    label: str


class LastFinishedOut(BaseModel):
    id: int
    config_path: str
    mode: str
    done_steps: int
    total_steps: int
    avg_loss: float | None = None
    status: str


class StartOptionsOut(BaseModel):
    start_from: dict[str, StartOptionOut]
    has_unfinished_run: bool
    last_finished: LastFinishedOut | None = None


def start_options_out(dto: StartOptionsResult) -> StartOptionsOut:
    return StartOptionsOut(
        start_from={
            key: StartOptionOut(path=opt.path, available=opt.available, label=opt.label)
            for key, opt in dto.start_from.items()
        },
        has_unfinished_run=dto.has_unfinished_run,
        last_finished=(
            None
            if dto.last_finished is None
            else LastFinishedOut(
                id=dto.last_finished.id,
                config_path=dto.last_finished.config_path,
                mode=dto.last_finished.mode,
                done_steps=dto.last_finished.done_steps,
                total_steps=dto.last_finished.total_steps,
                avg_loss=dto.last_finished.avg_loss,
                status=dto.last_finished.status,
            )
        ),
    )


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class SettingsViewOut(BaseModel):
    """Raw stored values and what they currently resolve to (a
    ``null`` resolved entry means nothing can resolve it)."""

    stored: dict[str, str]
    resolved: dict[str, str | None]


class SettingsIn(BaseModel):
    """Partial update: absent key = untouched, empty string = clear
    the override and fall back to auto-detection."""

    default_config: str | None = None
    comfy_dir: str | None = None
    venv_python: str | None = None
    checkpoints_dir: str | None = None
    loras_dir: str | None = None

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(
            default_config=self.default_config,
            comfy_dir=self.comfy_dir,
            venv_python=self.venv_python,
            checkpoints_dir=self.checkpoints_dir,
            loras_dir=self.loras_dir,
        )


def settings_out(view: SettingsView) -> SettingsViewOut:
    return SettingsViewOut(stored=dict(view.stored), resolved=dict(view.resolved))


# --------------------------------------------------------------------------
# Assets
# --------------------------------------------------------------------------


class AssetOptionOut(BaseModel):
    value: str
    label: str


class AssetCatalogOut(BaseModel):
    kind: str
    base_dir: str
    files: list[str]
    options: list[AssetOptionOut]
    upload_supported: bool
    browse_supported: bool


def asset_catalog_out(dto: AssetCatalog) -> AssetCatalogOut:
    return AssetCatalogOut(
        kind=dto.kind,
        base_dir=dto.base_dir,
        files=[option.value for option in dto.options],
        options=[AssetOptionOut(value=o.value, label=o.label) for o in dto.options],
        upload_supported=dto.upload_supported,
        browse_supported=dto.browse_supported,
    )


class AssetBrowseOut(BaseModel):
    kind: str
    path: str
    folders: list[str]
    files: list[str]


def asset_browse_out(dto: AssetBrowse) -> AssetBrowseOut:
    return AssetBrowseOut(
        kind=dto.kind,
        path=dto.path,
        folders=list(dto.folders),
        files=list(dto.files),
    )


class AssetPathOut(BaseModel):
    """Where a write landed (absolute path) for the client that sent
    a relative one."""

    kind: str
    relative_path: str
    path: str


# --------------------------------------------------------------------------
# Datasets (M3b) -- see docs/design/backend/04-dataset-format.md
# --------------------------------------------------------------------------


class DatasetInfoOut(BaseModel):
    name: str
    description: str
    created_at: datetime
    format_version: int


class DatasetStatsOut(BaseModel):
    items: int
    pending: int
    committed: int
    bad: int
    sets: int
    shards: int
    bytes: int


class DatasetSummaryOut(BaseModel):
    """List entry: identity always; ``stats`` is null for a legacy
    (pre-v2) dataset until it is migrated."""

    info: DatasetInfoOut
    stats: DatasetStatsOut | None = None


class DatasetListOut(BaseModel):
    datasets: list[DatasetSummaryOut]
    count: int


class DatasetItemOut(BaseModel):
    id: int
    source_id: int
    shard_id: int
    prompt: str
    neg_prompt: str
    model_type: str
    type: str
    cfg: float | None = None
    seed: int | None = None
    source_path: str | None = None
    latent_h: int
    latent_w: int
    preview_path: str | None = None
    committed: bool


class DatasetItemsOut(BaseModel):
    items: list[DatasetItemOut]
    count: int


class TrainingSetOut(BaseModel):
    id: int
    name: str
    description: str | None = None
    created_at: datetime
    members: int


class DatasetSetsOut(BaseModel):
    sets: list[TrainingSetOut]
    count: int


class DatasetTaskOut(BaseModel):
    id: int
    dataset: str
    kind: str
    status: str
    pid: int | None = None
    current: int
    total: int
    error: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime


class DatasetTasksOut(BaseModel):
    tasks: list[DatasetTaskOut]
    count: int


class DatasetDetailOut(BaseModel):
    info: DatasetInfoOut
    stats: DatasetStatsOut | None = None
    sets: list[TrainingSetOut]
    active_tasks: list[DatasetTaskOut]


class CreateDatasetIn(BaseModel):
    name: str = Field(min_length=1)
    description: str | None = None


class UpdateItemIn(BaseModel):
    """Partial edit: every ``null`` field is left untouched (a cleared
    caption is sent as ``""``, a real value)."""

    prompt: str | None = None
    neg_prompt: str | None = None
    cfg: float | None = None
    type: str | None = None


class BulkUpdateItemsIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)
    prompt: str | None = None
    prompt_mode: str = "set"
    neg_prompt: str | None = None
    cfg: float | None = None


class ItemIdsIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)


class CommitItemsIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)
    name: str = Field(min_length=1, description="training-set name")


class StartDatasetTaskIn(BaseModel):
    """Ingestion launch. ``model`` is a checkpoint path relative to the
    checkpoints dir; ``image_dir`` is an absolute server-side path."""

    kind: str = "ingest_lora"
    image_dir: str = Field(min_length=1)
    model: str = Field(min_length=1)
    recursive: bool = True
    resize_mode: str = "center_crop"
    latent_size: int = Field(default=64, ge=8)
    neg_prompt: str = ""
    model_type: str = "eps"
    seed: int = 42
    max_aspect_ratio: float = Field(default=2.0, ge=1.0)

    def to_command(self, dataset: str) -> StartDatasetTaskCommand:
        return StartDatasetTaskCommand(
            dataset=dataset,
            kind=self.kind,
            image_dir=self.image_dir,
            model=self.model,
            recursive=self.recursive,
            resize_mode=self.resize_mode,
            latent_size=self.latent_size,
            neg_prompt=self.neg_prompt,
            model_type=self.model_type,
            seed=self.seed,
            max_aspect_ratio=self.max_aspect_ratio,
        )


class DatasetDeletedOut(BaseModel):
    deleted: bool


class BulkUpdateOut(BaseModel):
    updated: int


class DiscardOut(BaseModel):
    deleted: int


class CommitOut(BaseModel):
    set_id: int
    set_name: str
    added: int


def dataset_info_out(info: DatasetInfo) -> DatasetInfoOut:
    return DatasetInfoOut(
        name=info.name,
        description=info.description,
        created_at=info.created_at,
        format_version=info.format_version,
    )


def dataset_stats_out(stats: DatasetStats) -> DatasetStatsOut:
    return DatasetStatsOut(
        items=stats.items,
        pending=stats.pending,
        committed=stats.committed,
        bad=stats.bad,
        sets=stats.sets,
        shards=stats.shards,
        bytes=stats.bytes,
    )


def dataset_summary_out(summary: DatasetSummary) -> DatasetSummaryOut:
    return DatasetSummaryOut(
        info=dataset_info_out(summary.info),
        stats=None if summary.stats is None else dataset_stats_out(summary.stats),
    )


def dataset_item_out(item: DatasetItem) -> DatasetItemOut:
    return DatasetItemOut(
        id=item.id,
        source_id=item.source_id,
        shard_id=item.shard_id,
        prompt=item.prompt,
        neg_prompt=item.neg_prompt,
        model_type=item.model_type,
        type=item.type,
        cfg=item.cfg,
        seed=item.seed,
        source_path=item.source_path,
        latent_h=item.latent_h,
        latent_w=item.latent_w,
        preview_path=item.preview_path,
        committed=item.committed,
    )


def training_set_out(info: TrainingSetInfo) -> TrainingSetOut:
    return TrainingSetOut(
        id=info.id,
        name=info.name,
        description=info.description,
        created_at=info.created_at,
        members=info.members,
    )


def dataset_task_out(task: DatasetTask) -> DatasetTaskOut:
    return DatasetTaskOut(
        id=task.id,
        dataset=task.dataset,
        kind=task.kind,
        status=task.status,
        pid=task.pid,
        current=task.current,
        total=task.total,
        error=task.error,
        params=dict(task.params),
        created_at=task.created_at,
        updated_at=task.updated_at,
    )


def dataset_detail_out(detail: DatasetDetail) -> DatasetDetailOut:
    return DatasetDetailOut(
        info=dataset_info_out(detail.info),
        stats=None if detail.stats is None else dataset_stats_out(detail.stats),
        sets=[training_set_out(info) for info in detail.sets],
        active_tasks=[dataset_task_out(t) for t in detail.active_tasks],
    )


def dataset_list_out(result) -> DatasetListOut:
    """``result``: application ``DatasetListResult``."""
    return DatasetListOut(
        datasets=[dataset_summary_out(s) for s in result.datasets], count=result.count
    )


def dataset_sets_out(infos: tuple[TrainingSetInfo, ...]) -> DatasetSetsOut:
    return DatasetSetsOut(
        sets=[training_set_out(info) for info in infos], count=len(infos)
    )


def dataset_tasks_out(result) -> DatasetTasksOut:
    """``result``: application ``DatasetTaskListResult``."""
    return DatasetTasksOut(
        tasks=[dataset_task_out(t) for t in result.tasks], count=result.count
    )
