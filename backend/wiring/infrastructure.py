"""The stateful base every other wiring step builds on: database, clock, events, settings, workspace, probes."""

from __future__ import annotations

from dataclasses import dataclass

from ..application.ports.clock import Clock
from ..application.event_publisher import EventPublisher
from ..config import Settings
from ..infrastructure.clock import SystemClock
from ..infrastructure.events.callback_event_bus import CallbackEventBus
from ..infrastructure.file_asset_store import FileSystemAssetStore
from ..infrastructure.graph.contract_catalog import model_environment
from ..infrastructure.monitor_bus import SharedMonitorBus
from ..infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from ..infrastructure.persistence.sqlite import SqliteDatabase
from ..application.limits import (
    DEVICE_REFRESH_MIN_SECONDS,
    READINESS_CACHE_SECONDS,
)
from ..application.ports.environment import (
    CachedDeviceProbe,
    TorchDeviceProbe,
)
from ..infrastructure.settings_store import SqliteSettingsStore
from ..infrastructure.workspace import WorkspaceLayout


def node_environment(layout):
    """What a node's resolve may ask for while a person edits: the model folders, as the run will see them."""
    from ..infrastructure.graph.tree import Environment  # noqa: PLC0415 -- the tree loads on first use

    try:
        return model_environment(layout.checkpoints_dir, layout.loras_dir, layout.datasets_dir)
    except RuntimeError:  # ComfyUI not found yet (first run): resolve still answers, with no files
        return Environment()


@dataclass(frozen=True, slots=True)
class Infrastructure:
    settings: Settings
    database: SqliteDatabase
    clock: Clock
    event_bus: CallbackEventBus
    publisher: EventPublisher
    settings_store: SqliteSettingsStore
    layout: WorkspaceLayout
    assets: FileSystemAssetStore
    executions: SqliteGraphExecutionRepository
    device_probe: CachedDeviceProbe
    monitor_bus: SharedMonitorBus

    @classmethod
    def build(cls, settings: Settings) -> "Infrastructure":
        database = SqliteDatabase(settings.db_path)
        database.initialize()

        event_bus = CallbackEventBus()
        clock = SystemClock()

        # Settings first: the workspace layout resolves its override tier through the store, and the store resolves
        # its reported view through the same policy (path_tiers) -- one definition each.
        settings_store = SqliteSettingsStore(database, settings.project_root)
        layout = WorkspaceLayout(settings.project_root, settings_kv=settings_store.get)

        executions = SqliteGraphExecutionRepository(database)

        # Wrapped, not bare: every device question in the server goes through here, and an unwrapped probe means one
        # torch import per request -- including from any web page the user has open, since a plain GET is
        # deliberately outside the Origin guard (ADR 0001).
        #
        # `is_busy` asks the execution repository rather than a supervisor flag, because the repository is what
        # already knows, and a second source of "is something running" is one that can disagree with the first.
        device_probe = CachedDeviceProbe(
            inner=TorchDeviceProbe(backend="xpu"),
            ttl=READINESS_CACHE_SECONDS,
            refresh_floor=DEVICE_REFRESH_MIN_SECONDS,
            is_busy=lambda: executions.find_active() is not None,
        )
        return cls(
            settings=settings,
            database=database,
            clock=clock,
            event_bus=event_bus,
            publisher=EventPublisher(events=event_bus),
            settings_store=settings_store,
            layout=layout,
            assets=FileSystemAssetStore(layout),
            executions=executions,
            device_probe=device_probe,
            # One bus per process: the runtime hands it to MonitorNode through ExecutionContext, the SSE endpoint
            # reads the same instance (M6).
            monitor_bus=SharedMonitorBus(),
        )
