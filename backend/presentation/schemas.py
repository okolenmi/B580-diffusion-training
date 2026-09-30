"""Response schemas (pydantic) -- the wire shape of this API.

One model per resource/operation, mapped explicitly from application
DTOs by the ``*_out`` helpers: the boundary is visible, and a DTO
change never silently reshapes responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ..application.dto import RunDTO, StartOptionsResult
from ..application.ports.asset_store import AssetCatalog, AssetBrowse
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
