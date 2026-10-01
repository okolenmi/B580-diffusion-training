"""Composition root -- the one module allowed to import every layer.

Order of business:

1. ``Settings`` arrive fully built (from the CLI);
2. infrastructure objects are constructed (database + migrations,
   repository, event bus, clock, settings store, workspace layout,
   training gateway, config inspector/files/options, assets store,
   artifacts, progress source);
3. the supervisor and use cases are constructed with those ports;
4. ``ReconcileRuns`` / ``ReconcileDatasetTasks`` sweep rows left
   unfinished by a previous process (before any request can race them);
5. the aggregate goes to ``presentation.create_app``.

Nothing else in the backend may know which concrete classes exist --
that is what makes swapping an implementation (or a fake in tests) a
one-line change here and nowhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .application.graph_supervisor import GraphExecutionSupervisor
from .application.ports.clock import Clock
from .application.ports.training_gateway import TrainingGateway
from .application.services import (
    ApplicationServices,
    AssetServices,
    ConfigServices,
    DatasetServices,
    GraphServices,
    SettingsServices,
)
from .application.supervisor import RunSupervisor
from .application.use_cases import (
    BrowseAssets,
    BulkUpdateDatasetItems,
    CommitDatasetItems,
    CreateDataset,
    DeleteDataset,
    DeleteGraph,
    DeleteGraphExecutions,
    DeleteRuns,
    DiscardDatasetItems,
    GetActiveRun,
    GetConfig,
    GetConfigOptions,
    GetDataset,
    GetGraph,
    GetGraphExecution,
    GetRun,
    GetRunLog,
    GetSettings,
    GetStartOptions,
    InspectAsset,
    ListAssets,
    ListDatasetItems,
    ListDatasetSets,
    ListDatasetTasks,
    ListDatasets,
    ListGraphExecutions,
    ListGraphs,
    ListNodeCatalog,
    ListRuns,
    MakeAssetFolder,
    NodeDiagnostics,
    ReadConfigRaw,
    ReadDatasetFile,
    ReconcileDatasetTasks,
    ReconcileGraphExecutions,
    ReconcileRuns,
    SaveGraph,
    SetDatasetPreview,
    StartDatasetTask,
    StartGraphExecution,
    StartTraining,
    StopDatasetTask,
    StopGraphExecution,
    StopTraining,
    UpdateConfig,
    UpdateDatasetItem,
    UpdateSettings,
    UploadAsset,
    ValidateGraph,
    WriteConfigRaw,
)
from .config import Settings
from .infrastructure.clock import SystemClock
from .infrastructure.config_options import PydanticConfigOptions
from .infrastructure.core_config_files import CoreConfigFiles
from .infrastructure.core_config_inspector import CoreConfigInspector
from .infrastructure.dataset_files import FsDatasetFiles
from .infrastructure.dataset_library import SqliteDatasetLibrary
from .infrastructure.dataset_previews import SqliteDatasetPreviews
from .infrastructure.dataset_task_gateway import SubprocessDatasetTaskGateway
from .infrastructure.dataset_tasks import SqliteDatasetTasks
from .infrastructure.directory_run_artifacts import DirectoryRunArtifacts
from .infrastructure.events.callback_event_bus import CallbackEventBus
from .infrastructure.file_asset_store import FileSystemAssetStore
from .infrastructure.graph.catalog import DiscoveredGraphCatalog
from .infrastructure.graph.discovery import NodeRegistry
from .infrastructure.graph.runtime import ReflectedGraphRuntime
from .infrastructure.jsonl_progress_source import JsonlProgressSource
from .infrastructure.monitor_bus import SharedMonitorBus
from .infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from .infrastructure.persistence.graph_library import SqliteGraphLibrary
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

    # A fresh database numbers runs from 1, but runs/run_<id>/ may already
    # hold a legacy run: the trainer opens its log with "w", so a
    # colliding id would truncate that history. Continue the sequence
    # above whatever is on disk instead (docs 07 F-04).
    legacy_high_water = artifacts.highest_existing_run_id()
    if legacy_high_water:
        run_repository.continue_ids_above(legacy_high_water)
        logger.info(
            "run ids continue above the existing runs/run_%d directory", legacy_high_water
        )

    # Dataset domain (M3b): library reads each dataset's own metadata.db,
    # task rows live in backend.db, and the fork gateway spawns children
    # through the same layout the settings resolve. The library is built
    # before the asset store so the catalog-only `dataset` kind can
    # list names through the same visibility rules as the API.
    dataset_library = SqliteDatasetLibrary(layout)
    dataset_tasks = SqliteDatasetTasks(database, clock)
    # Card preview pointer: backend.db row + first-item fallback read
    # through the library (M8f) -- server view state, never dataset files.
    dataset_previews = SqliteDatasetPreviews(database, dataset_library)
    dataset_gateway = SubprocessDatasetTaskGateway(layout, database.path)
    dataset_files = FsDatasetFiles(layout.datasets_dir)
    assets = FileSystemAssetStore(layout, datasets=dataset_library)

    # Graph domain (M4): one NodeRegistry feeds both the palette and the
    # executor -- a single discovery cache, swapped atomically on
    # refresh. Execution rows and the saved library are server state in
    # backend.db; the runtime keeps the default memory releaser (gc,
    # then the XPU caching allocator) so freed VRAM goes back to the
    # driver after every run.
    graph_registry = NodeRegistry()
    graph_catalog = DiscoveredGraphCatalog(graph_registry)
    # One bus per process: the runtime hands it to MonitorNode through
    # ExecutionContext, the SSE endpoint reads the same instance (M6).
    monitor_bus = SharedMonitorBus()
    graph_runtime = ReflectedGraphRuntime(graph_registry, monitor_bus=monitor_bus)
    graph_executions = SqliteGraphExecutionRepository(database)
    graph_library = SqliteGraphLibrary(database)
    graph_supervisor = GraphExecutionSupervisor(
        executions=graph_executions,
        runtime=graph_runtime,
        events=event_bus,
        clock=clock,
    )

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
            supervisor=supervisor,
            artifacts=artifacts,
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
        datasets=DatasetServices(
            list=ListDatasets(library=dataset_library, previews=dataset_previews),
            get=GetDataset(
                library=dataset_library,
                tasks=dataset_tasks,
                previews=dataset_previews,
            ),
            create=CreateDataset(library=dataset_library),
            delete=DeleteDataset(
                library=dataset_library,
                tasks=dataset_tasks,
                previews=dataset_previews,
            ),
            items=ListDatasetItems(library=dataset_library),
            update_item=UpdateDatasetItem(library=dataset_library),
            bulk_update=BulkUpdateDatasetItems(library=dataset_library),
            discard=DiscardDatasetItems(library=dataset_library),
            sets=ListDatasetSets(library=dataset_library),
            commit=CommitDatasetItems(library=dataset_library),
            tasks=ListDatasetTasks(
                library=dataset_library,
                tasks=dataset_tasks,
                gateway=dataset_gateway,
                clock=clock,
            ),
            start_task=StartDatasetTask(
                library=dataset_library,
                tasks=dataset_tasks,
                gateway=dataset_gateway,
                checkpoints_dir=layout.checkpoints_dir,
            ),
            stop_task=StopDatasetTask(
                tasks=dataset_tasks, gateway=dataset_gateway
            ),
            reconcile_tasks=ReconcileDatasetTasks(
                tasks=dataset_tasks, gateway=dataset_gateway
            ),
            read_file=ReadDatasetFile(files=dataset_files),
            set_preview=SetDatasetPreview(
                library=dataset_library, previews=dataset_previews
            ),
        ),
        graphs=GraphServices(
            catalog=ListNodeCatalog(catalog=graph_catalog),
            diagnostics=NodeDiagnostics(catalog=graph_catalog),
            validate=ValidateGraph(runtime=graph_runtime),
            start_execution=StartGraphExecution(
                executions=graph_executions,
                runtime=graph_runtime,
                events=event_bus,
                supervisor=graph_supervisor,
                clock=clock,
            ),
            list_executions=ListGraphExecutions(executions=graph_executions),
            get_execution=GetGraphExecution(executions=graph_executions),
            stop_execution=StopGraphExecution(
                executions=graph_executions,
                events=event_bus,
                supervisor=graph_supervisor,
                clock=clock,
            ),
            delete_executions=DeleteGraphExecutions(
                executions=graph_executions, events=event_bus
            ),
            reconcile_executions=ReconcileGraphExecutions(
                executions=graph_executions, events=event_bus, clock=clock
            ),
            save_graph=SaveGraph(library=graph_library),
            get_graph=GetGraph(library=graph_library),
            list_graphs=ListGraphs(library=graph_library),
            delete_graph=DeleteGraph(library=graph_library),
        ),
        # shared
        event_bus=event_bus,
        monitor_bus=monitor_bus,
    )

    # Startup sweep: nothing may observe an unfinished row from a dead
    # process once the server accepts requests.
    reconciled = services.reconcile_runs.execute()
    if reconciled.cleaned:
        logger.info("reconciled %d unfinished run(s) at startup", reconciled.cleaned)
    if reconciled.adopted:
        logger.info(
            "adopted %d still-training run(s) after the restart "
            "(their trainers were left running)",
            reconciled.adopted,
        )
    dataset_reconciled = services.datasets.reconcile_tasks.execute()
    if dataset_reconciled.cleaned:
        logger.info(
            "reconciled %d unfinished dataset task(s) at startup",
            dataset_reconciled.cleaned,
        )
    graph_reconciled = services.graphs.reconcile_executions.execute()
    if graph_reconciled.cleaned:
        logger.info(
            "reconciled %d unfinished graph execution(s) at startup",
            graph_reconciled.cleaned,
        )

    return Container(
        settings=settings,
        database=database,
        clock=clock,
        supervisor=supervisor,
        gateway=gateway,
        services=services,
    )
