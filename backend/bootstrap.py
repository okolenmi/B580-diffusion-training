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
import sys
from dataclasses import dataclass

from .application.graph_supervisor import GraphExecutionSupervisor
from .application.lifecycle_writer import ExecutionLifecycleWriter
from .application.memory_admission import LedgerProvider, release, task_owner
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
    InstallerServices,
    MonitorServices,
    SettingsServices,
)
from .application.use_cases import (
    ApplyInstallation,
    BrowseAssets,
    BulkUpdateDatasetItems,
    CheckComfyConflicts,
    GetInstall,
    InstallJob,
    StartInstall,
    CheckRequirements,
    CommitDatasetItems,
    CreateDataset,
    DeleteDataset,
    DeleteGraph,
    DeleteGraphExecutions,
    DescribeRequirements,
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
    SweepExecutionScratch,
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
from .config import GRAPH_EXECUTION_CHILD, Settings
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
from .infrastructure.graph_event_stream import ExecutionEventTail
from .infrastructure.graph_task_gateway import (
    InProcessGraphTaskGateway,
    SubprocessGraphTaskGateway,
)
from .application.ports.graph_task_gateway import GraphTaskGateway
from .application.ports.monitor_bus import MonitorBus
from .infrastructure.monitor_bus import SharedMonitorBus
from .infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from .infrastructure.persistence.graph_library import SqliteGraphLibrary
from .infrastructure.persistence.sqlite import SqliteDatabase
from .application.limits import (
    DEVICE_REFRESH_MIN_SECONDS,
    READINESS_CACHE_SECONDS,
)
from .application.ports.environment import (
    CachedDeviceProbe,
    MetadataPackageInventory,
    TorchDeviceProbe,
)
from .application.ports.comfy_environment import LocalComfyEnvironment
from .application.ports.package_installer import PipInstaller
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

    # Shared by readiness and by the wizard's GPU choice: one probe, one
    # answer, one 1.8s import.

    # Shared between the installer services so a job can be polled.
    install_jobs: dict[str, InstallJob] = {}

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

    # Wrapped, not bare: every device question in the server goes through
    # here, and an unwrapped probe means one torch import per request --
    # including from any web page the user has open, since a plain GET is
    # deliberately outside the Origin guard (ADR 0001).
    #
    # `is_busy` asks the execution repository rather than a supervisor flag,
    # because the repository is what already knows, and a second source of
    # "is something running" is one that can disagree with the first.
    device_probe = CachedDeviceProbe(
        inner=TorchDeviceProbe(backend="xpu"),
        ttl=READINESS_CACHE_SECONDS,
        refresh_floor=DEVICE_REFRESH_MIN_SECONDS,
        is_busy=lambda: graph_executions.find_active() is not None,
    )
    # One ledger per container, built on first use (MEM-03): the total
    # comes from the cached probe, which may only be able to answer
    # after the installer has run -- probing here would tax every
    # start-up and, worse, freeze "unknown" forever on a fresh machine.
    # Construction rebuilds it from the unfinished rows that carry a
    # claim, so a restart reproduces the same held total; the startup
    # reconcile below then releases the rows that turn out to be debris.
    memory_ledger = LedgerProvider(
        probe=device_probe,
        graph_executions=graph_executions,
        dataset_tasks=dataset_tasks,
        foreign_reserve_mb=settings.memory_foreign_reserve_mb,
        process_overhead_mb=settings.memory_process_overhead_mb,
    )
    # The task child finalises its own row through the WAL, so nothing
    # server-side would ever notice it ending: this waiter is what
    # hands the admission claim back when the child is actually gone.
    dataset_gateway = SubprocessDatasetTaskGateway(
        layout,
        database.path,
        on_exit=lambda task_id: release(memory_ledger, task_owner(task_id)),
    )
    task_sweeper = DatasetTaskSweeper(
        tasks=dataset_tasks,
        gateway=dataset_gateway,
        clock=clock,
        memory_ledger=memory_ledger,
    )
    graph_library = SqliteGraphLibrary(database)
    execution_writer = ExecutionLifecycleWriter(
        clock=clock,
        repository=graph_executions, events=publisher
    )
    # WP-22: the run happens somewhere other than here. Both gateways run
    # the *same* producer (graph_task_worker.run_execution) and both report
    # through the same event file, so choosing between them is a choice of
    # where the interpreter state lives and nothing else. The child gets
    # no registry -- it discovers its own, which costs a few seconds of
    # node discovery once per run and buys an isolation boundary the
    # server's already-warm registry cannot cross.
    if settings.graph_execution_mode == GRAPH_EXECUTION_CHILD:
        graph_gateway: GraphTaskGateway = SubprocessGraphTaskGateway(layout)
    else:
        graph_gateway = InProcessGraphTaskGateway(graph_registry)
    # Supervision scratch, next to the database rather than in the runs
    # dir: these are per-execution working files (the graph handed to the
    # child, its event stream, its log), not run artifacts anyone collects.
    graph_scratch = database.path.parent / "graph_executions"
    graph_scratch.mkdir(parents=True, exist_ok=True)
    # N3-07: nothing else ever removed these files, so the directory grew by
    # one run's worth forever while the UI reported the history as cleared.
    sweep_scratch = SweepExecutionScratch(
        executions=graph_executions, scratch_dir=graph_scratch,
    )
    graph_supervisor = GraphExecutionSupervisor(
        executions=graph_executions,
        writer=execution_writer,
        gateway=graph_gateway,
        events=publisher,
        clock=clock,
        memory_ledger=memory_ledger,
        monitor_bus=monitor_bus,
        scratch_dir=graph_scratch,
        make_tail=ExecutionEventTail,
    )

    services = ApplicationServices(
        config=ConfigServices(
            read=GetConfig(files=config_files, paths=paths),
            update=UpdateConfig(files=config_files, paths=paths),
            read_raw=ReadConfigRaw(files=config_files, paths=paths),
            write_raw=WriteConfigRaw(files=config_files, paths=paths),
            options=GetConfigOptions(options=config_options),
        ),
        installer=InstallerServices(
            # One probe object, used by both `check` and the devices route.
            # Two instances would each pay the 1.8s torch import; and a
            # wizard that reported the card from one probe and offered a
            # choice from another could show two different machines.
            check=CheckRequirements(
                inventory=MetadataPackageInventory(),
                device=device_probe,
                memory_ledger=memory_ledger,
            ),
            device_probe=device_probe,
            apply=ApplyInstallation(settings=settings_store),
            manifest=DescribeRequirements(),
            # No interpreter is fixed here. `venv_python` is a setting the
            # wizard sets, so the route resolves it and passes it per call --
            # a port that captured it at wiring time would report on
            # whichever venv happened to be configured when the server
            # started.
            conflicts=CheckComfyConflicts(environment=LocalComfyEnvironment()),
            # One job dict, shared: `install` writes it and `install_status`
            # reads it. The base interpreter is the one running the server,
            # which is the only interpreter known to work here.
            install=StartInstall(
                installer=PipInstaller(),
                project_root=settings.project_root,
                base_python=sys.executable,
                jobs=install_jobs,
                # Asked at install time rather than captured here: the wizard
                # exists to *find* this interpreter, and `execute` installs
                # into the one the server detects rather than the one a
                # request names. Same reason `conflicts` above takes the port
                # instead of a snapshot taken at wiring time.
                detect_comfy_python=lambda: LocalComfyEnvironment
                .default_venv_python(str(settings_store.get("comfy_dir", ""))),
            ),
            install_status=GetInstall(jobs=install_jobs),
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
                # A callable, so a checkpoints_dir changed in Settings is
                # used without a restart -- see StartDatasetTask.
                checkpoints_dir=lambda: layout.checkpoints_dir,
                memory_ledger=memory_ledger,
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
                memory_ledger=memory_ledger,
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
                executions=graph_executions, events=publisher,
                scratch=sweep_scratch,
                memory_ledger=memory_ledger,
            ),
            reconcile_executions=ReconcileGraphExecutions(
                executions=graph_executions,
                writer=execution_writer,
                launcher=graph_supervisor,
                clock=clock,
                memory_ledger=memory_ledger,
            ),
            save_graph=SaveGraph(library=graph_library),
            get_graph=GetGraph(library=graph_library),
            list_graphs=ListGraphs(library=graph_library),
            delete_graph=DeleteGraph(library=graph_library),
        ),
        # shared
        events=publisher,
        event_bus=event_bus,
        memory_ledger=memory_ledger,
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
    if graph_reconciled.still_running:
        # At warning rather than info, and here rather than only in the
        # use case's own log: this is the one startup state that is not
        # settled, and an operator who does not see it will find out by
        # trying to start a run and being told one is already active.
        logger.warning(
            "%d graph execution(s) left running at startup on a child that "
            "could not be adopted; their rows stay active on purpose, so the "
            "single-active check keeps refusing a second run. Each process "
            "is visible in `ps`.",
            graph_reconciled.still_running,
        )

    # After the reconcile, not before: adopting an unfinished run reads its
    # event file, so the sweep has to be able to see the difference between
    # "finished" and "still going", and that is what the reconcile has just
    # settled.
    scratch_swept = sweep_scratch.execute()
    if scratch_swept.files:
        logger.info(
            "removed scratch for %d finished graph execution(s) at startup: "
            "%d file(s), %.1f MB", scratch_swept.runs, scratch_swept.files,
            scratch_swept.bytes / 1_000_000,
        )

    return Container(
        settings=settings,
        database=database,
        clock=clock,
        graph_supervisor=graph_supervisor,
        services=services,
        monitor_bus=monitor_bus,
    )
