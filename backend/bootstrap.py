"""Composition root -- the one module allowed to import every layer.

Order of business:

1. ``Settings`` arrive fully built (from the CLI);
2. infrastructure objects are constructed (database + migrations,
   repository, event bus, clock, settings store, workspace layout,
   training gateway, config inspector/files/options, assets store,
   artifacts, progress source);
3. the supervisor and use cases are constructed with those ports;
4. ``ReconcileRuns`` sweeps rows left unfinished by a previous process
   (before any request can race it);
5. the aggregate goes to ``presentation.create_app``.

Nothing else in the backend may know which concrete classes exist --
that is what makes swapping an implementation (or a fake in tests) a
one-line change here and nowhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .application.ports.clock import Clock
from .application.ports.training_gateway import TrainingGateway
from .application.services import (
    ApplicationServices,
    AssetServices,
    ConfigServices,
    SettingsServices,
)
from .application.supervisor import RunSupervisor
from .application.use_cases import (
    BrowseAssets,
    DeleteRuns,
    GetActiveRun,
    GetConfig,
    GetConfigOptions,
    GetRun,
    GetRunLog,
    GetSettings,
    GetStartOptions,
    InspectAsset,
    ListAssets,
    ListRuns,
    MakeAssetFolder,
    ReadConfigRaw,
    ReconcileRuns,
    StartTraining,
    StopTraining,
    UpdateConfig,
    UpdateSettings,
    UploadAsset,
    WriteConfigRaw,
)
from .config import Settings
from .infrastructure.clock import SystemClock
from .infrastructure.config_options import PydanticConfigOptions
from .infrastructure.core_config_files import CoreConfigFiles
from .infrastructure.core_config_inspector import CoreConfigInspector
from .infrastructure.directory_run_artifacts import DirectoryRunArtifacts
from .infrastructure.events.callback_event_bus import CallbackEventBus
from .infrastructure.file_asset_store import FileSystemAssetStore
from .infrastructure.jsonl_progress_source import JsonlProgressSource
from .infrastructure.persistence.run_repository import SqliteRunRepository
from .infrastructure.persistence.sqlite import SqliteDatabase
from .infrastructure.settings_store import SqliteSettingsStore
from .infrastructure.subprocess_gateway import SubprocessTrainingGateway
from .infrastructure.workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Container:
    """Wired infrastructure + application objects for one process."""

    settings: Settings
    database: SqliteDatabase
    clock: Clock
    supervisor: RunSupervisor
    gateway: TrainingGateway
    services: ApplicationServices


def build_container(settings: Settings) -> Container:
    """Wire the real implementation graph (idempotent for the DB)."""
    database = SqliteDatabase(settings.db_path)
    database.initialize()

    run_repository = SqliteRunRepository(database)
    event_bus = CallbackEventBus()
    clock = SystemClock()

    # Settings first: the workspace layout resolves its override tier
    # through the store, and the store resolves its reported view
    # through the same policy (path_tiers) -- one definition each.
    settings_store = SqliteSettingsStore(database, settings.project_root)
    layout = WorkspaceLayout(
        settings.project_root, settings_kv=settings_store.get
    )
    artifacts = DirectoryRunArtifacts(layout)
    progress = JsonlProgressSource()
    gateway = SubprocessTrainingGateway(layout)
    inspector = CoreConfigInspector(layout)
    config_files = CoreConfigFiles()
    config_options = PydanticConfigOptions()
    assets = FileSystemAssetStore(layout)

    supervisor = RunSupervisor(
        runs=run_repository,
        events=event_bus,
        gateway=gateway,
        progress=progress,
        artifacts=artifacts,
        clock=clock,
    )

    services = ApplicationServices(
        list_runs=ListRuns(run_repository),
        get_run=GetRun(run_repository),
        delete_runs=DeleteRuns(run_repository, event_bus),
        get_active_run=GetActiveRun(run_repository),
        start_training=StartTraining(
            runs=run_repository,
            events=event_bus,
            gateway=gateway,
            inspector=inspector,
            artifacts=artifacts,
            supervisor=supervisor,
            clock=clock,
            project_root=settings.project_root,
        ),
        stop_training=StopTraining(
            runs=run_repository,
            events=event_bus,
            gateway=gateway,
            clock=clock,
        ),
        get_run_log=GetRunLog(runs=run_repository, artifacts=artifacts),
        reconcile_runs=ReconcileRuns(
            runs=run_repository,
            events=event_bus,
            gateway=gateway,
            clock=clock,
        ),
        config=ConfigServices(
            read=GetConfig(files=config_files, project_root=settings.project_root),
            update=UpdateConfig(files=config_files, project_root=settings.project_root),
            read_raw=ReadConfigRaw(files=config_files, project_root=settings.project_root),
            write_raw=WriteConfigRaw(files=config_files, project_root=settings.project_root),
            options=GetConfigOptions(options=config_options),
            start_options=GetStartOptions(
                inspector=inspector,
                runs=run_repository,
                project_root=settings.project_root,
            ),
        ),
        settings=SettingsServices(
            read=GetSettings(settings=settings_store),
            update=UpdateSettings(settings=settings_store),
        ),
        assets=AssetServices(
            list=ListAssets(assets=assets),
            browse=BrowseAssets(assets=assets),
            make_folder=MakeAssetFolder(assets=assets),
            upload=UploadAsset(assets=assets),
            inspect=InspectAsset(assets=assets),
        ),
        event_bus=event_bus,
    )

    # Startup sweep: nothing may observe an unfinished row from a dead
    # process once the server accepts requests.
    reconciled = services.reconcile_runs.execute()
    if reconciled.cleaned:
        logger.info("reconciled %d unfinished run(s) at startup", reconciled.cleaned)

    return Container(
        settings=settings,
        database=database,
        clock=clock,
        supervisor=supervisor,
        gateway=gateway,
        services=services,
    )
