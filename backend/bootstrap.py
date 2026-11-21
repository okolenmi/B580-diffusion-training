"""Composition root -- the one module allowed to import every layer.

Order of business:

1. ``Settings`` arrive fully built (from the CLI);
2. infrastructure objects are constructed (database + migrations,
   repository, event bus, clock, settings store, workspace layout,
   assets store, image-library and graph
   subsystems);
3. the supervisors and use cases are constructed with those ports;
4. ``ReconcileGraphExecutions`` sweeps rows
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

from .application.graph_peak_source import GraphPeakSource
from .application.graph_supervisor import GraphExecutionSupervisor
from .application.lifecycle_writer import ExecutionLifecycleWriter
from .application.memory_admission import LedgerProvider
from .application.ports.clock import Clock
from .application.event_publisher import EventPublisher
from .application.use_cases.app_crud import DeleteApp, DiffApp, GetApp, ListApps, SaveApp
from .application.use_cases.graph_modules import DescribeGraphModules, ResolveBoard
from .application.use_cases.run_app import RunApp
from .application.services import (
    ApplicationServices,
    AssetServices,
    GraphServices,
    LibraryServices,
    InstallerServices,
    MonitorServices,
    SettingsServices,
    AppServices,
)
from .application.use_cases import (
    ApplyInstallation,
    BrowseAssets,
    GetInstall,
    InstallJob,
    StartInstall,
    InspectEnvironment,
    TokenizerSetup,
    DeleteGraph,
    DeleteGraphExecutions,
    DescribeRequirements,
    GetGraph,
    GetExecutionMetrics,
    GetGraphExecution,
    GetSettings,
    InspectAsset,
    ListAssets,
    ListGraphExecutions,
    ListGraphs,
    MakeAssetFolder,
    NodeDiagnostics,
    ReconcileGraphExecutions,
    SweepExecutionScratch,
    SaveGraph,
    StartGraphExecution,
    SubscribeMonitor,
    RenameGraphExecution,
    SendExecutionAction,
    StopGraphExecution,
    UpdateSettings,
    UploadAsset,
    PreviewGraphMemory,
    ValidateGraph,
)
from .config import GRAPH_EXECUTION_CHILD, Settings
from .infrastructure.server_restart import ExecServerRestarter
from .infrastructure.clock import SystemClock
from .infrastructure.file_artifact_store import FileArtifactStore
from .infrastructure.file_command_channel import FileCommandChannel
from .infrastructure.host_memory import ProcHostMemory
from .infrastructure.image_libraries import FsImageLibraries
from .infrastructure.events.callback_event_bus import CallbackEventBus
from .infrastructure.file_asset_store import FileSystemAssetStore
from .application.use_cases.image_libraries import (
    AddToSet, CreateLibrary, DeleteCache, DeleteLibrary, DeleteSet, EditCaptions, LibraryReport, ListImageIds, ListImages, ListImports,
    ListLibraries, ReadImage, RemoveImages, StartImport,
)
from .application.use_cases.node_shapes import GetNodeCatalog, ListImageLibraries, ListModelFiles, ResolveNode
from .application.use_cases.starters import BuildStarter, ListStarters
from .infrastructure.graph.contract_catalog import ContractNodeCatalog, model_environment
from .infrastructure.graph.next_starters import NextStarters
from .infrastructure.graph.memory_fields import memory_fields_resolver
from .infrastructure.graph.runtime import ContractGraphRuntime
from .infrastructure.execution_metrics import FileExecutionMetrics
from .infrastructure.graph_event_stream import ExecutionEventTail
from .infrastructure.graph_task_gateway import (
    InProcessGraphTaskGateway,
    SubprocessGraphTaskGateway,
)
from .application.ports.graph_task_gateway import GraphTaskGateway
from .application.ports.monitor_bus import MonitorBus
from .infrastructure.memory_peak_store import SqlitePeakStore
from .infrastructure.monitor_bus import SharedMonitorBus
from .infrastructure.persistence.graph_execution_repository import (
    SqliteGraphExecutionRepository,
)
from .infrastructure.persistence.graph_library import SqliteGraphLibrary
from .infrastructure.persistence.app_library import SqliteAppLibrary
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
from .application.ports.tokenizer_files import HubTokenizerFiles
from .application.ports.venv_inspection import LocalVenvInspector
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


def _node_environment(layout):
    """What a node's resolve may ask for while a person edits: the model folders, as the run will see them."""
    from .infrastructure.graph.tree import Environment  # noqa: PLC0415 -- the tree loads on first use

    try:
        return model_environment(layout.checkpoints_dir, layout.loras_dir, layout.datasets_dir)
    except RuntimeError:  # ComfyUI not found yet (first run): resolve still answers, with no files
        return Environment()


def build_container(settings: Settings) -> Container:
    """Wire the real implementation graph (idempotent for the DB)."""
    database = SqliteDatabase(settings.db_path)
    database.initialize()

    event_bus = CallbackEventBus()
    publisher = EventPublisher(events=event_bus)
    clock = SystemClock()

    # Settings first: the workspace layout resolves its override tier
    # through the store, and the store resolves its reported view
    # through the same policy (path_tiers) -- one definition each.
    settings_store = SqliteSettingsStore(database, settings.project_root)
    layout = WorkspaceLayout(
        settings.project_root, settings_kv=settings_store.get
    )

    # Shared by memory sizing and by the wizard's GPU list: one probe, one
    # answer, one 1.8s import.

    # Shared between the installer services so a job can be polled.
    install_jobs: dict[str, InstallJob] = {}

    assets = FileSystemAssetStore(layout)

    # Graph domain (M4): one NodeRegistry feeds both the palette and the
    # executor -- a single discovery cache, swapped atomically on
    # refresh. Execution rows and the saved library are server state in
    # backend.db; the runtime keeps the default memory releaser (gc,
    # then the XPU caching allocator) so freed VRAM goes back to the
    # driver after every run.
    node_catalog = ContractNodeCatalog(environment=lambda: _node_environment(layout))
    graph_registry = node_catalog.registry
    # One bus per process: the runtime hands it to MonitorNode through
    # ExecutionContext, the SSE endpoint reads the same instance (M6).
    monitor_bus = SharedMonitorBus()
    graph_runtime = ContractGraphRuntime(graph_registry, env=lambda: _node_environment(layout))
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
        foreign_reserve_mb=settings.memory_foreign_reserve_mb,
        process_overhead_mb=settings.memory_process_overhead_mb,
    )
    graph_library = SqliteGraphLibrary(database)
    app_library = SqliteAppLibrary(database)
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
        graph_gateway = InProcessGraphTaskGateway(
            graph_registry, environment=lambda: _node_environment(layout)
        )
    # Supervision scratch, next to the database rather than in the runs
    # dir: these are per-execution working files (the graph handed to the
    # child, its event stream, its log), not run artifacts anyone collects.
    graph_scratch = database.path.parent / "graph_executions"
    graph_scratch.mkdir(parents=True, exist_ok=True)
    # N3-07: nothing else ever removed these files, so the directory grew by
    # one run's worth forever while the UI reported the history as cleared.
    # What a run reported to its monitors outlives that scratch: the sweep files it first.
    execution_metrics = FileExecutionMetrics(graph_scratch)
    sweep_scratch = SweepExecutionScratch(
        executions=graph_executions, scratch_dir=graph_scratch, metrics=execution_metrics,
    )
    # Remembered peaks per configuration (MEM-04), in their own file
    # next to the database: one writer for peaks, shared by admission
    # (reads through it) and the watcher (files what a child reports).
    # Self-creating table, so no migration owns it.
    peak_store = SqlitePeakStore(database.path.parent / "memory_peaks.db")
    graph_supervisor = GraphExecutionSupervisor(
        executions=graph_executions,
        writer=execution_writer,
        gateway=graph_gateway,
        events=publisher,
        clock=clock,
        memory_ledger=memory_ledger,
        peak_store=peak_store,
        monitor_bus=monitor_bus,
        scratch_dir=graph_scratch,
        make_tail=ExecutionEventTail,
    )

    # Fingerprint (MEM-01) + remembered peak (MEM-04), built once here
    # and shared by everything that has to agree on which configuration
    # a peak belongs to: the panel's preview, the run admission below,
    # and the watcher above that files this run's own reported peaks.
    # Three constructions of the same thing is three chances not to.
    graph_peak_source = GraphPeakSource(
        peaks=peak_store,
        resolve_memory_fields=memory_fields_resolver(graph_registry),
    )

    # One instance, shared by the editor's Run and by running an app:
    # the single-active check lives under this object's lock, so a second
    # instance would be a second lock and a window in which two graphs
    # could both pass it.
    start_execution = StartGraphExecution(
        executions=graph_executions,
        writer=execution_writer,
        runtime=graph_runtime,
        launcher=graph_supervisor,
        clock=clock,
        memory_ledger=memory_ledger,
        # The same instance the panel previews through: the fingerprint
        # it computes is what the watcher above files this run's own
        # reported peaks under, and a preview keyed differently would
        # quote a peak belonging to some other configuration.
        peak_source=graph_peak_source.observed,
        host_memory=ProcHostMemory(),
    )
    describe_modules = DescribeGraphModules(catalog=node_catalog)
    resolve_board = ResolveBoard(describe=describe_modules, runtime=graph_runtime)

    image_libraries = FsImageLibraries(layout)
    inspect_environment = InspectEnvironment(LocalVenvInspector(), settings.project_root)
    services = ApplicationServices(
        installer=InstallerServices(
            device_probe=device_probe,
            apply=ApplyInstallation(settings=settings_store, project_root=settings.project_root, restarter=ExecServerRestarter()),
            manifest=DescribeRequirements(),
            inspect=inspect_environment,
            # One job dict, shared: `install` writes it and `install_status` reads it. New environments are made with the interpreter
            # running the server, the only one known to work here.
            install=StartInstall(
                installer=PipInstaller(),
                inspect=inspect_environment,
                base_python=getattr(sys, "_base_executable", sys.executable),
                jobs=install_jobs,
            ),
            install_status=GetInstall(jobs=install_jobs),
            tokenizers=TokenizerSetup(HubTokenizerFiles(settings.project_root)),
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
        libraries=LibraryServices(
            list=ListLibraries(image_libraries), create=CreateLibrary(image_libraries), delete=DeleteLibrary(image_libraries),
            images=ListImages(image_libraries), image_ids=ListImageIds(image_libraries), report=LibraryReport(image_libraries), read_image=ReadImage(image_libraries), edit_captions=EditCaptions(image_libraries),
            remove_images=RemoveImages(image_libraries), add_to_set=AddToSet(image_libraries), delete_set=DeleteSet(image_libraries),
            delete_cache=DeleteCache(image_libraries), start_import=StartImport(image_libraries), imports=ListImports(image_libraries),
        ),
        graphs=GraphServices(
            node_catalog=GetNodeCatalog(catalog=node_catalog),
            resolve_node=ResolveNode(catalog=node_catalog),
            model_files=ListModelFiles(catalog=node_catalog),
            image_libraries=ListImageLibraries(catalog=node_catalog),
            list_starters=ListStarters(starters=NextStarters()),
            build_starter=BuildStarter(starters=NextStarters()),
            diagnostics=NodeDiagnostics(catalog=node_catalog),
            validate=ValidateGraph(runtime=graph_runtime),
            preview_memory=PreviewGraphMemory(
                memory_ledger=memory_ledger,
                peak_source=graph_peak_source.observed,
            ),
            start_execution=start_execution,
            list_executions=ListGraphExecutions(executions=graph_executions),
            get_execution=GetGraphExecution(executions=graph_executions),
            execution_metrics=GetExecutionMetrics(executions=graph_executions, metrics=execution_metrics),
            rename_execution=RenameGraphExecution(executions=graph_executions),
            send_action=SendExecutionAction(
                executions=graph_executions, catalog=node_catalog, channel=FileCommandChannel(graph_scratch)),
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
        artifacts=FileArtifactStore(layout.artifacts_dir),
        apps=AppServices(
            describe=describe_modules,
            resolve=resolve_board,
            save=SaveApp(library=app_library),
            get=GetApp(library=app_library),
            list=ListApps(library=app_library),
            delete=DeleteApp(library=app_library),
            diff=DiffApp(library=app_library, resolve=resolve_board),
            run=RunApp(library=app_library, start=start_execution),
        ),
    )

    # Startup sweep: nothing may observe an unfinished row from a dead
    # process once the server accepts requests.
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
