"""Response schemas (pydantic) -- the wire shape of this API.

One model per resource/operation, mapped explicitly from application
DTOs by the ``*_out`` helpers: the boundary is visible, and a DTO
change never silently reshapes responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..application.dto import (
    DatasetDetail,
    GraphExecutionDTO,
    GraphExecutionSummaryDTO,
    StartDatasetTaskCommand,
    TeacherTaskParams,
)
from ..application.ports.asset_store import AssetCatalog, AssetBrowse
from ..application.ports.graph_catalog import CatalogSnapshot, NodeInfo, PortInfo
from ..application.ports.graph_runtime import GraphIssue
from ..domain.graph import GRAPH_FORMAT, GraphDefinition, GraphEdgeSpec, GraphNodeSpec
from ..domain.memory_settings import AUTO, MemorySettings
from ..application.ports.dataset_library import (
    DatasetInfo,
    DatasetItem,
    DatasetStats,
    DatasetSummary,
    TrainingSetInfo,
)
from ..application.ports.dataset_tasks import DatasetTask, TaskKind
from ..application.ports.settings_store import SettingsChanges, SettingsView
from ..application.use_cases.apply_installation import InstallationState
from ..application.use_cases.check_comfy_conflicts import (
    CONFLICT,
    SAFE,
    UNKNOWN,
    ConflictReport,
)
from ..application.use_cases.check_requirements import ReadinessReport
from ..domain.value_objects import GraphStatus


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str
    version: str

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
    models_dir: str | None = None

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(
            default_config=self.default_config,
            comfy_dir=self.comfy_dir,
            venv_python=self.venv_python,
            checkpoints_dir=self.checkpoints_dir,
            loras_dir=self.loras_dir,
            models_dir=self.models_dir,
        )


def settings_out(view: SettingsView) -> SettingsViewOut:
    return SettingsViewOut(stored=dict(view.stored), resolved=dict(view.resolved))


# --------------------------------------------------------------------------
# Installer (first run)
# --------------------------------------------------------------------------


class InstallerPackageOut(BaseModel):
    """One manifest row and whether this machine has it.

    `why` is on the row rather than in the wizard, so the sentence a user
    reads about a missing package is written next to the requirement it
    describes and cannot drift from it.
    """

    name: str
    tier: str
    why: str
    installed: bool
    installed_version: str | None
    blocking: bool
    approx_mb: int | None
    never_install: bool


class InstallerReadinessOut(BaseModel):
    ready: bool
    packages: list[InstallerPackageOut]
    missing: list[str]
    #: False when the package check already answered the question and the
    #: device was never asked. A client must not render "no device" for a
    #: machine nobody asked.
    device_checked: bool
    device_present: bool
    device_name: str | None
    device_total_memory_mb: float | None
    device_reason: str | None
    device_detail: str | None
    #: What a full install would fetch, counting only installable absences.
    approx_download_mb: int


class InstallerStateOut(BaseModel):
    configured: bool
    resolved_comfy_dir: str | None
    resolved_checkpoints_dir: str | None
    resolved_loras_dir: str | None
    missing: list[str]


class ConflictFindingOut(BaseModel):
    """One package in ComfyUI's venv, and which of the three states it is in.

    `outcome` is sent as the string rather than collapsed into a boolean,
    because the three states get three different treatments and a boolean
    would make the client re-derive the distinction from two fields.
    """

    name: str
    installed_version: str | None
    declared_specifier: str | None
    outcome: str
    description: str


class AdditionFindingOut(BaseModel):
    """One of our four packages, against ComfyUI's declarations."""

    name: str
    declared_specifier: str | None
    blocked: bool
    description: str


class InstallerDeviceOut(BaseModel):
    """One card. `index` is torch's device index, which is what a user
    passes to `torch.xpu.set_device` and therefore what the choice has to
    refer to -- not the position in this list."""

    index: int
    present: bool
    name: str | None
    total_memory_mb: float | None
    reason: str | None


class InstallerDevicesOut(BaseModel):
    """Every card, and whether anybody looked.

    `enumerated` is the same distinction as `ReadinessReport.device_checked`:
    an empty list from a probe that ran and found nothing is not the same
    answer as one from a probe that failed, and a wizard offering a GPU
    choice needs to be able to tell them apart.
    """

    enumerated: bool
    backend: str
    devices: list[InstallerDeviceOut]
    #: Why there are none, when there are none. None when there are some.
    reason: str | None


class InstallerInstallIn(BaseModel):
    """What to install, and where.

    `constraints` is sent by the client rather than recomputed server-side
    because the server must never be handed a weaker pin set than the one it
    would build -- but the conflict check is a 0.16s subprocess in another
    interpreter and re-running it here would double that and make the two
    answers differ. So the pins are computed once, by the endpoint that can
    be trusted to compute them, and this request says which target they
    belong to.
    """

    target: str
    packages: list[str]
    constraints: list[str] = Field(default_factory=list)
    #: Only for `target="comfy"`. Which interpreter to install into.
    comfy_venv_python: str | None = None


class InstallerInstallOut(BaseModel):
    """The job, immediately. Never the result.

    The response exists so the client has an id to poll, which is why it
    carries the command and the pins: a user agreeing to an install that
    writes into a virtualenv they did not create should be able to read
    what will run before it runs.
    """

    id: str
    state: str
    target_label: str
    packages: list[str]
    constraints: list[str]
    command: list[str]
    log: list[str]
    error: str | None
    target_python: str | None
    terminal: bool


class InstallerConflictsOut(BaseModel):
    """May the four server packages go into ComfyUI's venv?

    `checked` is separate from `safe` on purpose. "We could not check" and
    "we checked and it is fine" must never render the same, and a client
    that only received `safe` would have to guess which one it was.
    """

    comfy_dir: str
    checked: bool
    safe: bool
    refusal_reason: str | None
    requirements_path: str | None
    venv_python: str | None
    requirements_error: str | None
    venv_error: str | None
    counts: dict[str, int]
    #: Every finding, ordered: conflicts first, because a refusal must lead
    #: with the reason for it.
    findings: list[ConflictFindingOut]
    additions: list[AdditionFindingOut]
    #: The exact pins an install would use. Sent so the wizard can show what
    #: "nothing already installed can change" concretely means -- 185 lines
    #: on this machine -- rather than asking for that on trust.
    constraints: list[str]


def conflicts_out(report: ConflictReport) -> InstallerConflictsOut:
    order = {CONFLICT: 0, UNKNOWN: 1, SAFE: 2}
    return InstallerConflictsOut(
        comfy_dir=report.comfy_dir,
        checked=report.checked,
        safe=report.safe,
        refusal_reason=report.refusal_reason(),
        requirements_path=report.requirements_path,
        venv_python=report.venv_python,
        requirements_error=report.requirements_error,
        venv_error=report.venv_error,
        counts={
            "total": len(report.findings),
            "safe": len([f for f in report.findings if f.outcome == SAFE]),
            "conflict": len(report.conflicts),
            "unknown": len(report.unknowns),
        },
        findings=[
            ConflictFindingOut(
                name=f.name,
                installed_version=f.installed_version,
                declared_specifier=f.declared_specifier,
                outcome=f.outcome,
                description=f.describe(),
            )
            for f in sorted(report.findings, key=lambda f: (order[f.outcome], f.name))
        ],
        additions=[
            AdditionFindingOut(
                name=a.name,
                declared_specifier=a.declared_specifier,
                blocked=a.blocked,
                description=a.describe(),
            )
            for a in report.additions
        ],
        constraints=list(report.constraints),
    )


class InstallerApplyIn(BaseModel):
    """The paths the wizard collected. All optional -- an absent key is
    left alone, so a user who accepts every default sends none of them."""

    comfy_dir: str | None = None
    checkpoints_dir: str | None = None
    loras_dir: str | None = None
    models_dir: str | None = None

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(
            comfy_dir=self.comfy_dir,
            checkpoints_dir=self.checkpoints_dir,
            loras_dir=self.loras_dir,
            models_dir=self.models_dir,
        )


class InstallerApplyOut(BaseModel):
    """The fresh settings view plus the new state, in one response.

    Both, because "did that finish the wizard" is the question the client
    asks next and answering it from the server's own post-write read is
    what stops the wizard guessing.
    """

    settings: dict[str, Any]
    state: dict[str, Any]


def readiness_out(report: ReadinessReport) -> InstallerReadinessOut:
    return InstallerReadinessOut(
        ready=report.ready,
        packages=[
            InstallerPackageOut(
                name=row.name,
                tier=row.requirement.tier,
                why=row.requirement.why,
                installed=row.satisfied,
                installed_version=row.installed_version,
                blocking=row.blocking,
                approx_mb=row.requirement.approx_mb,
                never_install=row.requirement.never_install,
            )
            for row in report.packages
        ],
        missing=[row.name for row in report.missing],
        device_checked=report.device_checked,
        device_present=report.device_present,
        device_name=report.device_name,
        device_total_memory_mb=report.device_total_memory_mb,
        device_reason=report.device_reason,
        device_detail=report.device_detail,
        approx_download_mb=report.approx_download_mb,
    )


def state_out(state: InstallationState) -> InstallerStateOut:
    return InstallerStateOut(
        configured=state.configured,
        resolved_comfy_dir=state.resolved_comfy_dir,
        resolved_checkpoints_dir=state.resolved_checkpoints_dir,
        resolved_loras_dir=state.resolved_loras_dir,
        missing=list(state.missing),
    )


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
    (pre-v2) dataset until it is migrated. ``preview_path`` is the
    resolved card image (stored override or first non-bad item's
    preview), null when the dataset has none."""

    info: DatasetInfoOut
    stats: DatasetStatsOut | None = None
    preview_path: str | None = None


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
    """Rows in *this page* -- not the size of the result."""
    # The window actually served (docs 07 F-14, docs 08 Q10). `limit` is
    # always populated now: there is no "everything" window any more,
    # because dataset size is unbounded by ingestion.
    limit: int | None = None
    offset: int = 0
    total: int | None = None
    """Rows matching the filter in the whole dataset."""
    next_offset: int | None = None
    """Offset of the next page; null when this is the last one."""


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
    preview_path: str | None = None
    sets: list[TrainingSetOut]
    active_tasks: list[DatasetTaskOut]


class SetPreviewIn(BaseModel):
    """``PUT /{name}/preview`` body: the item whose image fronts the
    card. An id, never a path -- the server reads the path itself."""

    item_id: int = Field(ge=1)


class DatasetPreviewOut(BaseModel):
    preview_path: str


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
    prompt_mode: str = Field("set", description="set | prepend | append")
    neg_prompt: str | None = None
    neg_prompt_mode: str = Field("set", description="set | prepend | append")
    cfg: float | None = None
    type: str | None = Field(None, description="good | bad (multi-edit verdict)")


class ItemIdsIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)


class CommitItemsIn(BaseModel):
    item_ids: list[int] = Field(min_length=1)
    name: str = Field(min_length=1, description="training-set name")


class StartDatasetTaskIn(BaseModel):
    """Task launch, one body for both kinds (M8e).

    ``kind: ingest_lora`` consumes the import fields -- ``model`` is a
    checkpoint path relative to the checkpoints dir, ``image_dir`` an
    absolute server-side path. ``kind: generate_teacher`` consumes the
    teacher fields below (flat, mirroring the legacy ``type="teacher"``
    body); ``seed``/``latent_size``/``model_type`` are shared by both.
    Semantic validation (modes, ranges, prompt content, per-kind
    required fields) is the use case's job -- ``kind`` stays a plain
    string so an unknown one leaves as ``invalid_query``, not a
    schema-level ``validation_error``.
    """

    kind: str = "ingest_lora"
    image_dir: str = Field(
        "",
        description="required for kind=ingest_lora: absolute server-side "
        "source directory; unused by generate_teacher",
    )
    model: str = Field(min_length=1)
    recursive: bool = True
    resize_mode: str = "center_crop"
    latent_size: int = Field(default=64, ge=8)
    neg_prompt: str = ""
    model_type: str = "eps"
    seed: int = 42
    max_aspect_ratio: float = Field(default=2.0, ge=1.0)

    # -- generate_teacher options (legacy type="teacher") -----------------
    prompt_mode: str = "list"  # list | keywords
    prompts: str = Field("", description="newline-separated prompts (mode=list)")
    keywords: str = Field("", description="newline-separated keywords (mode=keywords)")
    keywords_file: str = Field("", description="optional server-side .txt/.csv word list")
    template: str = Field("", description='prompt template, "{keywords}" placeholder')
    min_keywords: int = Field(3, ge=1)
    max_keywords: int = Field(10, ge=1)
    neg_mode: str = "list"  # list | keywords
    negative_prompt: str = Field("", description="one negative string for every sample")
    neg_keywords: str = ""
    neg_keywords_file: str = ""
    neg_template: str = ""
    neg_min_keywords: int = Field(3, ge=1)
    neg_max_keywords: int = Field(10, ge=1)
    cfg_min: float = 3.0
    cfg_max: float = 9.0
    steps_min: int = Field(20, ge=1)
    steps_max: int = Field(30, ge=1)
    t_mode: str = "uniform"  # uniform | low | mid | high | logit
    t_low: int = Field(20, ge=0)
    t_high: int = Field(999, ge=0)
    batch_size: int = Field(1, ge=1, le=256)
    n_conditions: int = Field(10, ge=1)
    n_samples_per_cond: int = Field(1, ge=1)

    def to_command(self, dataset: str) -> StartDatasetTaskCommand:
        teacher: TeacherTaskParams | None = None
        # Compared by value, never coerced: the wire type is a plain str
        # precisely so an unknown kind stays a *use case* 422 with the
        # vocabulary in the message, instead of a ValueError here (the
        # enum is str-valued, so this is the same comparison).
        if self.kind == TaskKind.GENERATE_TEACHER.value:
            teacher = TeacherTaskParams(
                prompt_mode=self.prompt_mode,
                prompts=self.prompts,
                keywords=self.keywords,
                keywords_file=self.keywords_file,
                template=self.template,
                min_keywords=self.min_keywords,
                max_keywords=self.max_keywords,
                neg_mode=self.neg_mode,
                negative_prompt=self.negative_prompt,
                neg_keywords=self.neg_keywords,
                neg_keywords_file=self.neg_keywords_file,
                neg_template=self.neg_template,
                neg_min_keywords=self.neg_min_keywords,
                neg_max_keywords=self.neg_max_keywords,
                cfg_min=self.cfg_min,
                cfg_max=self.cfg_max,
                steps_min=self.steps_min,
                steps_max=self.steps_max,
                t_mode=self.t_mode,
                t_low=self.t_low,
                t_high=self.t_high,
                batch_size=self.batch_size,
                seed=self.seed,
                n_conditions=self.n_conditions,
                n_samples_per_cond=self.n_samples_per_cond,
                latent_size=self.latent_size,
                model_type=self.model_type,
            )
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
            teacher=teacher,
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
        preview_path=summary.preview_path,
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
        preview_path=detail.preview_path,
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


# ---------------------------------------------------------------------------
# Graph domain (M4) -- docs/design/backend/05-graph-runtime.md section 6
# ---------------------------------------------------------------------------


class GraphNodeIn(BaseModel):
    """One submitted node. ``id``/``class_name`` are *not* constrained
    here: empty/duplicate ids and unknown classes are graph validation
    issues (complete, localized, listed together) -- a 422 body rejection
    would truncate that report to one field error."""

    id: str
    class_name: str
    params: dict[str, Any] = Field(default_factory=dict)


class GraphEdgeIn(BaseModel):
    from_node: str
    from_port: str
    to_node: str
    to_port: str


class MemorySettingsIn(BaseModel):
    """Per-graph memory settings on a submission.

    Unlike the storage decoder -- which is tolerant by design because it
    reads our own snapshots -- this is the caller-facing edge: an unknown
    setting or a value that is neither a number nor ``"auto"`` is a 422
    naming the offender, not silently defaulted.
    """

    model_config = ConfigDict(extra="forbid")

    vram_min_mb: float = 0.0
    vram_max_mb: float | str = AUTO
    strict: bool = True
    policy: str = "demand_driven"
    ram_max_mb: float | str = AUTO

    @field_validator("vram_max_mb", "ram_max_mb")
    @classmethod
    def _number_or_auto(cls, value: Any) -> Any:
        if isinstance(value, str) and value != AUTO:
            raise ValueError(f"{value!r} is not a number or {AUTO!r}")
        return value

    def to_domain(self) -> MemorySettings:
        return MemorySettings(
            vram_min_mb=self.vram_min_mb,
            vram_max_mb=self.vram_max_mb,
            strict=self.strict,
            policy=self.policy,
            ram_max_mb=self.ram_max_mb,
        )


class GraphRunIn(BaseModel):
    """Submission body for ``/validate`` and ``/run``."""

    nodes: list[GraphNodeIn]
    edges: list[GraphEdgeIn] = Field(default_factory=list)
    memory: MemorySettingsIn | None = None

    def to_definition(self) -> GraphDefinition:
        return GraphDefinition(
            nodes=tuple(
                GraphNodeSpec(id=n.id, class_name=n.class_name, params=dict(n.params))
                for n in self.nodes
            ),
            edges=tuple(
                GraphEdgeSpec(
                    from_node=e.from_node,
                    from_port=e.from_port,
                    to_node=e.to_node,
                    to_port=e.to_port,
                )
                for e in self.edges
            ),
            memory=self.memory.to_domain() if self.memory else MemorySettings(),
        )


class DiagnosticsIn(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class PortOut(BaseModel):
    """One declared port: JSON ``default`` *and* ``default_repr`` so
    consumers never have to parse reprs (and never lose non-JSON
    defaults). Outputs carry the identity defaults (no default, no
    hints)."""

    name: str
    type: str
    required: bool
    doc: str = ""
    type_mro: list[str] = Field(default_factory=list)
    default: Any = None
    default_repr: str | None = None
    path_kind: str | None = None
    choices: list[str] | None = None
    visible_when: list[Any] | None = None
    widget_only: bool = False


class PresetOut(BaseModel):
    name: str
    required_inputs: list[PortOut]
    required_outputs: list[PortOut]


class GraphNodeOut(BaseModel):
    """One palette entry, reflected from the real class."""

    class_name: str
    display_name: str
    domain: str
    module: str
    doc: str = ""
    bases: list[str]
    inputs: list[PortOut]
    outputs: list[PortOut]
    node_kind: str
    presets: list[PresetOut] | None = None
    has_diagnostics: bool = False


class CatalogLoadErrorOut(BaseModel):
    module: str
    message: str


class GraphCatalogOut(BaseModel):
    count: int
    domains: dict[str, list[GraphNodeOut]]
    load_errors: list[CatalogLoadErrorOut]


class DiagnosticsOut(BaseModel):
    messages: dict[str, list[str]]


class GraphIssueOut(BaseModel):
    severity: str
    code: str
    message: str
    node_id: str | None = None
    edge_index: int | None = None
    param: str | None = None


class ValidateOut(BaseModel):
    """Always 200: ``ok=false`` means the run endpoint would reject
    this submission (422 with these same issues)."""

    ok: bool
    issues: list[GraphIssueOut]


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
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExecutionOut(ExecutionSummaryOut):
    results: list[NodeResultOut]
    graph: dict[str, Any]


class ExecutionListOut(BaseModel):
    executions: list[ExecutionSummaryOut]
    count: int


class DeleteExecutionsOut(BaseModel):
    deleted: int


class LibraryGraphIn(BaseModel):
    """``PUT /library/{name}``: a graph payload plus an optional
    description. Unknown top-level keys ride along verbatim (the
    library stores what was submitted; validation happens at run)."""

    model_config = ConfigDict(extra="allow")

    format: int | None = None
    nodes: list[GraphNodeIn] = Field(default_factory=list)
    edges: list[GraphEdgeIn] = Field(default_factory=list)
    description: str = ""

    def to_payload(self) -> dict:
        payload = dict(self.model_extra or {})
        payload["format"] = self.format if self.format is not None else GRAPH_FORMAT
        payload["nodes"] = [n.model_dump() for n in self.nodes]
        payload["edges"] = [e.model_dump() for e in self.edges]
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


def port_out(port: PortInfo) -> PortOut:
    return PortOut(
        name=port.name,
        type=port.type,
        required=port.required,
        doc=port.doc,
        type_mro=list(port.type_mro),
        default=port.default,
        default_repr=port.default_repr,
        path_kind=port.path_kind,
        choices=list(port.choices) if port.choices is not None else None,
        visible_when=list(port.visible_when) if port.visible_when is not None else None,
        widget_only=port.widget_only,
    )


def graph_node_out(info: NodeInfo) -> GraphNodeOut:
    presets = (
        None
        if info.presets is None
        else [
            PresetOut(
                name=preset.name,
                required_inputs=[port_out(p) for p in preset.required_inputs],
                required_outputs=[port_out(p) for p in preset.required_outputs],
            )
            for preset in info.presets
        ]
    )
    return GraphNodeOut(
        class_name=info.class_name,
        display_name=info.display_name,
        domain=info.domain,
        module=info.module,
        doc=info.doc,
        bases=list(info.bases),
        inputs=[port_out(p) for p in info.inputs],
        outputs=[port_out(p) for p in info.outputs],
        node_kind=info.node_kind,
        presets=presets,
        has_diagnostics=info.has_diagnostics,
    )


def graph_catalog_out(snapshot: CatalogSnapshot) -> GraphCatalogOut:
    return GraphCatalogOut(
        count=len(snapshot.nodes),
        domains={
            domain: [graph_node_out(node) for node in nodes]
            for domain, nodes in snapshot.domains.items()
        },
        load_errors=[
            CatalogLoadErrorOut(module=e.module, message=e.message)
            for e in snapshot.load_errors
        ],
    )


def graph_issue_out(issue: GraphIssue) -> GraphIssueOut:
    return GraphIssueOut(
        severity=issue.severity,
        code=issue.code,
        message=issue.message,
        node_id=issue.node_id,
        edge_index=issue.edge_index,
        param=issue.param,
    )


def execution_summary_out(dto: GraphExecutionSummaryDTO) -> ExecutionSummaryOut:
    return ExecutionSummaryOut(
        execution_id=dto.execution_id,
        status=dto.status,
        error=dto.error,
        created_at=dto.created_at,
        updated_at=dto.updated_at,
        started_at=dto.started_at,
        finished_at=dto.finished_at,
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
    )


def execution_list_out(result) -> ExecutionListOut:
    """``result``: application ``ExecutionListResult``."""
    return ExecutionListOut(
        executions=[execution_summary_out(e) for e in result.executions],
        count=result.count,
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
