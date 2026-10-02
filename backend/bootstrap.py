"""Composition root -- the one module allowed to import every layer.

Order of business:

1. ``Settings`` arrive fully built (from the CLI);
2. infrastructure objects are constructed (database + migrations,
   repository, event bus, clock, settings store, workspace layout,
   config inspector/files/options, assets store, dataset and graph
   subsystems);
3. the supervisors and use cases are constructed with those ports;
4. ``ReconcileDatasetTasks`` / ``ReconcileGraphExecutions`` sweep rows
   left unfinished by a previous process (before any request can race
   them);
5. the aggregate goes to ``presentation.create_app``.

Nothing else in the backend may know which concrete classes exist --
that is what makes swapping an implementation (or a fake in tests) a
one-line change here and nowhere else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .application.graph_supervisor import GraphExecutionSupervisor
from .application.lifecycle_writer import ExecutionLifecycleWriter
from .application.project_paths import ProjectPaths
from .application.ports.clock import Clock
from .application.dataset_task_sweeper import DatasetTaskSweeper
from .application.event_publisher import EventPublisher
from .application.services import (
    ApplicationServices,
    AssetServices,
    ConfigServices,
    DatasetServices,
    GraphServices,
    MonitorServices,
    SettingsServices,
)
from .application.use_cases import (
    BrowseAssets,
    BulkUpdateDatasetItems,
    CommitDatasetItems,
    CreateDataset,
    DeleteDataset,
    DeleteGraph,
    DeleteGraphExecutions,
    DiscardDatasetItems,
    GetConfig,
    GetConfigOptions,
    GetDataset,
    GetGraph,
    GetGraphExecution,
    GetSettings,
    InspectAsset,
    ListAssets,
    ListDatasetItems,
    ListDatasetSets,
    ListDatasetTasks,
    ListDatasets,
    ListGraphExecutions,
    ListGraphs,
    ListNodeCatalog,
    MakeAssetFolder,
    NodeDiagnostics,
    ReadConfigRaw,
    ReadDatasetFile,
    ReconcileDatasetTasks,
    ReconcileGraphExecutions,
    SaveGraph,
    SetDatasetPreview,
    StartDatasetTask,
    StartGraphExecution,
    SubscribeMonitor,
    StopDatasetTask,
    StopGraphExecution,
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
from .infrastructure.dataset_files import FsDatasetFiles
from .infrastructure.dataset_library import SqliteDatasetLibrary
from .infrastructure.dataset_previews import SqliteDatasetPreviews
from .infrastructure.dataset_task_gateway import SubprocessDatasetTaskGateway
from .infrastructure.dataset_tasks import SqliteDatasetTasks
from .infrastructure.events.callback_event_bus import CallbackEventBus
from .infrastructure.file_asset_store import FileSystemAssetStore
from .infrastructure.graph.catalog import DiscoveredGraphCatalog
from .infrastructure.graph.discovery import NodeRegistry
from .infrastructure.graph.runtime import ReflectedGraphRuntime
from .application.ports.monitor_bus import MonitorBus
from .infrastructure.monitor_bus import SharedMonitorBus
from .infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from .infrastructure.persistence.graph_library import SqliteGraphLibrary
from .infrastructure.persistence.sqlite import SqliteDatabase
from .infrastructure.settings_store import SqliteSettingsStore
from .infrastructure.workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Container:
    """Wired infrastructure + application objects for one process."""

    settings: Settings
    database: SqliteDatabase
    clock: Clock
    graph_supervisor: GraphExecutionSupervisor
    services: ApplicationServices
    # The monitor adapter stays reachable for the few callers that
    # legitimately *publish* telemetry (tests; the graph runtime's
    # ExecutionContext wiring). Subscribing goes through
    # ``services.monitor.subscribe`` -- see docs 08 S-04.
    monitor_bus: MonitorBus


def build_container(settings: Settings) -> Container:
    """Wire the real implementation graph (idempotent for the DB)."""
    database = SqliteDatabase(settings.db_path)
    database.initialize()

    event_bus = CallbackEventBus()
    publisher = EventPublisher(events=event_bus)
    paths = ProjectPaths(root=settings.project_root)
    clock = SystemClock()

    # Settings first: the workspace layout resolves its override tier
    # through the store, and the store resolves its reported view
    # through the same policy (path_tiers) -- one definition each.
    settings_store = SqliteSettingsStore(database, settings.project_root)
    layout = WorkspaceLayout(
        settings.project_root, settings_kv=settings_store.get
    )
    config_files = CoreConfigFiles()
    config_options = PydanticConfigOptions()

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
    task_sweeper = DatasetTaskSweeper(
        tasks=dataset_tasks, gateway=dataset_gateway, clock=clock
    )
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
    execution_writer = ExecutionLifecycleWriter(
        repository=graph_executions, events=publisher
    )
    graph_supervisor = GraphExecutionSupervisor(
        executions=graph_executions,
        writer=execution_writer,
        runtime=graph_runtime,
        events=publisher,
        clock=clock,
    )

    services = ApplicationServices(
        config=ConfigServices(
            read=GetConfig(files=config_files, paths=paths),
            update=UpdateConfig(files=config_files, paths=paths),
            read_raw=ReadConfigRaw(files=config_files, paths=paths),
            write_raw=WriteConfigRaw(files=config_files, paths=paths),
            options=GetConfigOptions(options=config_options),
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
            ),
            start_task=StartDatasetTask(
                library=dataset_library,
                tasks=dataset_tasks,
                gateway=dataset_gateway,
                checkpoints_dir=layout.checkpoints_dir,
                sweeper=task_sweeper,
            ),
            stop_task=StopDatasetTask(
                tasks=dataset_tasks, gateway=dataset_gateway
            ),
            reconcile_tasks=ReconcileDatasetTasks(sweeper=task_sweeper),
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
                writer=execution_writer,
                runtime=graph_runtime,
                launcher=graph_supervisor,
                clock=clock,
            ),
            list_executions=ListGraphExecutions(executions=graph_executions),
            get_execution=GetGraphExecution(executions=graph_executions),
            stop_execution=StopGraphExecution(
                executions=graph_executions,
                writer=execution_writer,
                launcher=graph_supervisor,
                clock=clock,
            ),
            delete_executions=DeleteGraphExecutions(
                executions=graph_executions, events=publisher
            ),
            reconcile_executions=ReconcileGraphExecutions(
                executions=graph_executions,
                writer=execution_writer,
                clock=clock,
            ),
            save_graph=SaveGraph(library=graph_library),
            get_graph=GetGraph(library=graph_library),
            list_graphs=ListGraphs(library=graph_library),
            delete_graph=DeleteGraph(library=graph_library),
        ),
        # shared
        events=publisher,
        event_bus=event_bus,
        monitor=MonitorServices(subscribe=SubscribeMonitor(bus=monitor_bus)),
    )

    # Startup sweep: nothing may observe an unfinished row from a dead
    # process once the server accepts requests.
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
        graph_supervisor=graph_supervisor,
        services=services,
        monitor_bus=monitor_bus,
    )
