"""Response schemas (pydantic) -- the wire shape of this API.

One model per resource/operation, mapped explicitly from application
DTOs by the ``*_out`` helpers: the boundary is visible, and a DTO
change never silently reshapes responses.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..application.dto import GraphExecutionDTO, GraphExecutionSummaryDTO
from ..application.ports.asset_store import AssetCatalog, AssetBrowse
from ..application.ports.graph_runtime import GraphIssue
from ..domain.graph import GRAPH_FORMAT, GraphDefinition
from ..domain.memory_settings import AUTO
from ..application.ports.settings_store import SettingsChanges, SettingsView
from ..application.use_cases.apply_installation import InstallationState
from ..application.ports.requirements_manifest import TORCH_TESTED_VERSION
from ..application.use_cases.inspect_environment import EnvironmentReport
from ..domain.value_objects import GraphStatus


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class HealthOut(BaseModel):
    status: str
    version: str
    #: The admission ledger's snapshot (MEM-03): capacity, holders and
    #: their sizes, what is free. None while the device total is
    #: unknown and no ledger exists.
    memory: dict | None = None

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

    comfy_dir: str | None = None
    venv_python: str | None = None
    checkpoints_dir: str | None = None
    loras_dir: str | None = None
    models_dir: str | None = None

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(
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


class InstallerStateOut(BaseModel):
    configured: bool
    resolved_checkpoints_dir: str | None
    resolved_loras_dir: str | None
    resolved_models_dir: str | None
    #: Where the model folder goes when the user accepts the default.
    default_models_dir: str
    missing: list[str]
    #: The latest install job, running or finished, so a reloaded setup page can resume it.
    install_id: str | None = None


class InstallerDeviceOut(BaseModel):
    """One card. ``index`` is torch's device index, not the position in the list."""

    index: int
    present: bool
    name: str | None
    total_memory_mb: float | None
    reason: str | None
    #: Set when the row did not come from torch (see ``SystemCards``).
    note: str | None = None


class InstallerDevicesOut(BaseModel):
    """Every card, and whether anybody looked: an empty list from a probe that ran differs from one that failed."""

    enumerated: bool
    backend: str
    devices: list[InstallerDeviceOut]
    reason: str | None


class PackageStateOut(BaseModel):
    name: str
    tier: str
    why: str
    approx_mb: int | None
    installed_version: str | None


class InstallerEnvironmentOut(BaseModel):
    """What an environment holds, and what installing into it would do."""

    kind: str
    venv_dir: str
    interpreter: str | None
    python_version: str | None
    error: str | None
    packages: list[PackageStateOut]
    to_install: list[str]
    download_mb: int
    installed_count: int
    problems: list[str]
    #: What the install deletes first: only ever the project's own virtualenv.
    discards: list[str] = []
    usable: bool
    #: The torch release offered as the default.
    tested_torch: str


class InstallerVerifyOut(BaseModel):
    present: bool
    name: str | None
    total_memory_mb: int | None
    torch_version: str | None
    reason: str | None


class InstallerInstallIn(BaseModel):
    """Where to install. What gets installed is decided by the server from the environment itself."""

    target: Literal["new", "existing"]
    #: The virtualenv folder; required for ``existing``.
    venv_dir: str | None = None
    #: Which torch to install when the environment has none: ``tested`` (the release the trainer is tested with), ``latest``
    #: (what the Intel XPU index offers) or an exact version such as ``2.11.0``.
    torch: str = "tested"


class InstallerInstallOut(BaseModel):
    """The job. ``log`` is the tail of pip's output and ``target_python`` is set once it succeeded."""

    id: str
    state: str
    target_label: str
    packages: list[str]
    log: list[str]
    error: str | None
    target_python: str | None
    terminal: bool


class InstallerApplyIn(BaseModel):
    """What the wizard collected; an absent key is left alone."""

    models_dir: str | None = None
    venv_python: str | None = None

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(models_dir=self.models_dir, venv_python=self.venv_python)


class InstallerApplyOut(BaseModel):
    """The fresh settings view plus the new state, so the client learns whether setup is finished from the server's own read."""

    settings: dict[str, Any]
    state: dict[str, Any]
    #: True when the server is restarting into the chosen environment; the page waits for /health before leaving.
    restarting: bool = False
    #: Set when the server could not move into the chosen environment, and why.
    restart_note: str | None = None


def environment_out(report: EnvironmentReport) -> InstallerEnvironmentOut:
    return InstallerEnvironmentOut(
        kind=report.kind,
        venv_dir=report.venv_dir,
        interpreter=report.interpreter,
        python_version=report.python_version,
        error=report.error,
        packages=[PackageStateOut(name=p.name, tier=p.tier, why=p.why, approx_mb=p.approx_mb, installed_version=p.installed_version)
                  for p in report.packages],
        to_install=list(report.to_install),
        download_mb=report.download_mb,
        installed_count=report.installed_count,
        problems=list(report.problems),
        discards=list(report.discards),
        usable=report.usable,
        tested_torch=TORCH_TESTED_VERSION,
    )


def state_out(state: InstallationState, install: dict | None = None) -> InstallerStateOut:
    return InstallerStateOut(
        install_id=install["id"] if install else None,
        configured=state.configured,
        resolved_checkpoints_dir=state.resolved_checkpoints_dir,
        resolved_loras_dir=state.resolved_loras_dir,
        resolved_models_dir=state.resolved_models_dir,
        default_models_dir=state.default_models_dir,
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


# ---------------------------------------------------------------------------
# Graph domain (M4) -- archive/old-documentation/design/backend/05-graph-runtime.md section 6
# ---------------------------------------------------------------------------


class MemoryOverridesIn(BaseModel):
    """Per-execution overrides of the graph's memory settings.

    All-optional: anything not named keeps the graph's value. Unknown
    keys are rejected (``extra="forbid"``) with the offending key in
    the 422 -- a typo in an override must not silently become "use the
    graph's default", because the caller asked to change exactly that
    value.

    MEM-05: ``device`` is deliberately not one of the overridable keys.
    The child runs the device the *graph file* names; an override would
    live only on the row, which the child never sees -- so the key is
    refused here and the domain's merge drops it too, rather than
    being accepted into a value nothing could honour.

    MEM-03H-01: the same number checks as ``MemorySettings`` -- an
    override arrives through the same endpoint and would otherwise be
    the softer way into the ledger with a bad value.
    """

    model_config = ConfigDict(extra="forbid")

    vram_min_mb: float | None = Field(None, ge=0, allow_inf_nan=False)
    vram_max_mb: float | str | None = None
    strict: bool | None = None
    policy: str | None = None
    ram_max_mb: float | str | None = None

    @field_validator("vram_min_mb", "vram_max_mb", "ram_max_mb", mode="before")
    @classmethod
    def _no_boolean_budget(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{value!r} is a boolean, not a memory budget")
        return value

    @field_validator("vram_max_mb", "ram_max_mb", mode="before")
    @classmethod
    def _number_or_auto(cls, value: Any) -> Any:
        # None means "keep the graph's value" and must fall through to
        # the union; everything else is a value the caller is changing.
        if value is None:
            return value
        if isinstance(value, str):
            if value != AUTO:
                raise ValueError(f"{value!r} is not a number or {AUTO!r}")
            return value
        if isinstance(value, (int, float)):
            try:
                number = float(value)
            except OverflowError:
                # An int past the float range (10**400): pydantic turns
                # ValueError into the 422, but would let OverflowError
                # escape as a 500.
                raise ValueError(
                    f"{value!r} is too large to be a device size"
                ) from None
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{value!r} is not a finite number above zero")
            return value
        return value

    @model_validator(mode="after")
    def _min_within_max(self) -> MemoryOverridesIn:
        maximum = self.vram_max_mb
        if (
            self.vram_min_mb is not None
            and isinstance(maximum, float)
            and self.vram_min_mb > maximum
        ):
            raise ValueError(
                f"vram_min_mb ({self.vram_min_mb}) exceeds vram_max_mb ({maximum})"
            )
        return self

    def as_overrides(self) -> dict:
        """Only the keys that were named -- absent means "keep the
        graph's value", and a present key carries its override value."""
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None
        }


class GraphRunIn(BaseModel):
    """Submission body for ``/validate`` and ``/run``: the graph document (docs/contracts/graph-json.md), whose
    ``settings`` carry the memory budget, and this run's own overrides of it."""

    graph: dict[str, Any]
    memory_overrides: MemoryOverridesIn | None = None
    #: What a start does when the card has no room: ``parallel`` refuses, ``auto`` plans the run, ``queue`` plans it
    #: whenever anything else lives or waits (``/validate`` ignores it).
    mode: Literal["auto", "parallel", "queue"] = "auto"
    #: What the run is called in the list (blank: none).
    name: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def _graph_reads_and_override_stays_in_range(self) -> GraphRunIn:
        """The graph must be a document this version reads, and the *merged* memory request must be ordered, not just
        each half: an override that lifts the floor above the graph's ceiling would otherwise reach the use case and
        die there as a 500 (domain errors escaping a use case are always a bug), instead of here as a 422 naming
        both numbers."""
        memory = self.to_definition().memory
        if self.memory_overrides is None:
            return self
        minimum = (
            self.memory_overrides.vram_min_mb
            if self.memory_overrides.vram_min_mb is not None
            else memory.vram_min_mb
        )
        maximum = (
            self.memory_overrides.vram_max_mb
            if self.memory_overrides.vram_max_mb is not None
            else memory.vram_max_mb
        )
        if not isinstance(maximum, str) and minimum > maximum:
            raise ValueError(
                f"vram_min_mb ({minimum}) exceeds vram_max_mb ({maximum}) "
                "with the overrides applied"
            )
        return self

    def to_definition(self) -> GraphDefinition:
        return GraphDefinition.from_dict(self.graph)


class DiagnosticsIn(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class DiagnosticsOut(BaseModel):
    messages: dict[str, list[str]]


class GraphIssueOut(BaseModel):
    severity: str
    code: str
    message: str
    node_id: str | None = None
    edge_index: int | None = None
    param: str | None = None
    plane: str = "data"


class MemoryPreviewOut(BaseModel):
    """What a run of this graph would need, and whether it fits now.

    Always 200 -- a preview never refuses anything, it reports. The
    nulls are the point: ``peak_mb`` null is "never measured", not
    "needs nothing", and ``device_demand_mb`` null is "no ledger yet".
    """

    fingerprint_key: str | None
    peak_mb: float | None
    demand_mb: float | None
    demand_source: str
    exploratory: bool
    device_demand_mb: float | None
    verdict: str
    reason: str | None
    capacity_mb: float | None
    free_mb: float | None
    held_mb: float | None
    foreign_reserve_mb: float | None
    holders: dict[str, dict]


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
    name: str | None = None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None


class ExecutionOut(ExecutionSummaryOut):
    results: list[NodeResultOut]
    graph: dict[str, Any]
    memory: dict[str, Any] | None = None


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


class LibraryGraphIn(BaseModel):
    """``PUT /library/{name}``: a graph payload plus an optional
    description. Unknown top-level keys ride along verbatim (the
    library stores what was submitted; validation happens at run)."""

    model_config = ConfigDict(extra="allow")

    format: int | None = None
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    description: str = ""

    def to_payload(self) -> dict:
        payload = dict(self.model_extra or {})
        payload["format"] = self.format if self.format is not None else GRAPH_FORMAT
        payload["nodes"] = list(self.nodes)
        payload["edges"] = list(self.edges)
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


def graph_issue_out(issue: GraphIssue) -> GraphIssueOut:
    return GraphIssueOut(
        severity=issue.severity,
        code=issue.code,
        message=issue.message,
        node_id=issue.node_id,
        edge_index=issue.edge_index,
        param=issue.param,
        plane=issue.plane,
    )


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


# --------------------------------------------------------------------------
# Templates (saved pages / monitor windows) -- docs/design/15

class TemplateIn(BaseModel):
    """``PUT /templates/{name}``: the page document plus a description.

    ``doc`` is deliberately an open object here: its shape is checked by
    ``domain/template.validate_template_doc`` (which reports *every*
    problem with a path), not by a pydantic model that would stop at the
    first and could not carry the keys a newer frontend adds.
    """

    description: str = ""
    doc: dict[str, Any]


class SavedTemplateOut(BaseModel):
    name: str
    description: str = ""
    kind: str
    doc: dict[str, Any]
    block_count: int
    node_count: int
    created_at: datetime
    updated_at: datetime


class SavedTemplateSummaryOut(BaseModel):
    name: str
    description: str = ""
    kind: str
    block_count: int
    node_count: int
    created_at: datetime
    updated_at: datetime


class SavedTemplateListOut(BaseModel):
    templates: list[SavedTemplateSummaryOut]
    count: int


class DeleteTemplateOut(BaseModel):
    deleted: bool


class ModulesIn(BaseModel):
    """``POST /templates/modules``: a graph document (the editor's payload shape)."""

    graph: dict[str, Any]

    def to_definition(self) -> GraphDefinition:
        return GraphDefinition.from_dict(self.graph)


class GraphModulesOut(BaseModel):
    """Everything a page could bind to. The per-node and per-module dicts
    are open on purpose -- they mirror ``nodes/ui.py`` descriptors, which
    grow without a schema release."""

    nodes: list[dict[str, Any]]
    graph_fields: list[dict[str, Any]]
    graph_displays: list[dict[str, Any]]


class ResolveIn(BaseModel):
    """``POST /templates/resolve``: a draft document, optionally with the
    graph to resolve a monitor window against."""

    doc: dict[str, Any]
    graph: dict[str, Any] | None = None


class BlockStatusOut(BaseModel):
    id: str
    status: str
    detail: str = ""


class ResolveOut(BaseModel):
    modules: GraphModulesOut
    blocks: list[BlockStatusOut]
    issues: list[dict[str, Any]]


class TemplateRunIn(BaseModel):
    """``POST /templates/{name}/run``.

    ``values`` is ``{"nodes": {node_id: {port: value}}, "memory": {key:
    value}}`` and may only name fields the page exposes."""

    values: dict[str, Any] = Field(default_factory=dict)
    memory_overrides: MemoryOverridesIn | None = None


def saved_template_out(dto) -> SavedTemplateOut:
    return SavedTemplateOut(
        name=dto.name, description=dto.description, kind=dto.kind, doc=dto.doc,
        block_count=dto.block_count, node_count=dto.node_count,
        created_at=dto.created_at, updated_at=dto.updated_at,
    )


def saved_template_list_out(result) -> SavedTemplateListOut:
    return SavedTemplateListOut(
        templates=[
            SavedTemplateSummaryOut(
                name=t.name, description=t.description, kind=t.kind,
                block_count=t.block_count, node_count=t.node_count,
                created_at=t.created_at, updated_at=t.updated_at,
            )
            for t in result.templates
        ],
        count=result.count,
    )


def graph_modules_out(modules) -> GraphModulesOut:
    return GraphModulesOut(
        nodes=[dict(n) for n in modules.nodes],
        graph_fields=[dict(f) for f in modules.graph_fields],
        graph_displays=[dict(d) for d in modules.graph_displays],
    )


def resolve_out(resolution) -> ResolveOut:
    return ResolveOut(
        modules=graph_modules_out(resolution.modules),
        blocks=[BlockStatusOut(**b) for b in resolution.blocks],
        issues=[dict(i) for i in resolution.issues],
    )
