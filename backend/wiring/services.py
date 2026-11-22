"""The application services, one builder per group, from the two kits."""

from __future__ import annotations

import sys

from ..application.use_cases.app_crud import DeleteApp, DiffApp, GetApp, ListApps, SaveApp
from ..application.use_cases.run_app import RunApp
from ..application.services import (
    ApplicationServices,
    AssetServices,
    GraphServices,
    LibraryServices,
    InstallerServices,
    MonitorServices,
    SettingsServices,
    AppServices,
)
from ..application.use_cases import (
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
    SaveGraph,
    SubscribeMonitor,
    RenameGraphExecution,
    SendExecutionAction,
    StopGraphExecution,
    UpdateSettings,
    UploadAsset,
    PreviewGraphMemory,
    ValidateGraph,
)
from ..infrastructure.server_restart import ExecServerRestarter
from ..infrastructure.file_artifact_store import FileArtifactStore
from ..infrastructure.file_command_channel import FileCommandChannel
from ..infrastructure.image_libraries import FsImageLibraries
from ..infrastructure.file_asset_store import FileSystemAssetStore
from ..application.use_cases.image_libraries import (
    AddToSet, CreateLibrary, DeleteCache, DeleteLibrary, DeleteSet, EditCaptions, LibraryReport, ListImageIds, ListImages, ListImports,
    ListLibraries, ReadImage, RemoveImages, StartImport,
)
from ..application.use_cases.node_shapes import GetNodeCatalog, ListImageLibraries, ListModelFiles, ResolveNode
from ..application.use_cases.starters import BuildStarter, ListStarters
from ..infrastructure.graph.next_starters import NextStarters
from ..infrastructure.hub_tokenizer_files import HubTokenizerFiles
from ..infrastructure.local_venv_inspector import LocalVenvInspector
from ..infrastructure.pip_installer import PipInstaller

from .graph_kit import GraphKit
from .infrastructure import Infrastructure


def build_services(infra: Infrastructure, kit: GraphKit) -> ApplicationServices:
    return ApplicationServices(
        installer=_installer(infra),
        settings=SettingsServices(
            read=GetSettings(settings=infra.settings_store),
            update=UpdateSettings(settings=infra.settings_store),
        ),
        assets=_assets(infra.assets),
        libraries=_libraries(FsImageLibraries(infra.layout)),
        graphs=_graphs(infra, kit),
        events=infra.publisher,
        event_bus=infra.event_bus,
        memory_ledger=kit.memory_ledger,
        monitor=MonitorServices(subscribe=SubscribeMonitor(bus=infra.monitor_bus)),
        artifacts=FileArtifactStore(infra.layout.artifacts_dir),
        apps=_apps(kit),
    )


def _installer(infra: Infrastructure) -> InstallerServices:
    root = infra.settings.project_root
    inspect_environment = InspectEnvironment(LocalVenvInspector(), root)
    # One job dict, shared: `install` writes it and `install_status` reads it. New environments are made with the
    # interpreter running the server, the only one known to work here.
    install_jobs: dict[str, InstallJob] = {}
    return InstallerServices(
        device_probe=infra.device_probe,
        apply=ApplyInstallation(settings=infra.settings_store, project_root=root, restarter=ExecServerRestarter()),
        manifest=DescribeRequirements(),
        inspect=inspect_environment,
        install=StartInstall(
            installer=PipInstaller(),
            inspect=inspect_environment,
            base_python=getattr(sys, "_base_executable", sys.executable),
            jobs=install_jobs,
        ),
        install_status=GetInstall(jobs=install_jobs),
        tokenizers=TokenizerSetup(HubTokenizerFiles(root)),
    )


def _assets(assets: FileSystemAssetStore) -> AssetServices:
    return AssetServices(
        list=ListAssets(assets=assets),
        browse=BrowseAssets(assets=assets),
        make_folder=MakeAssetFolder(assets=assets),
        upload=UploadAsset(assets=assets),
        inspect=InspectAsset(assets=assets),
    )


def _libraries(images: FsImageLibraries) -> LibraryServices:
    return LibraryServices(
        list=ListLibraries(images),
        create=CreateLibrary(images),
        delete=DeleteLibrary(images),
        images=ListImages(images),
        image_ids=ListImageIds(images),
        report=LibraryReport(images),
        read_image=ReadImage(images),
        edit_captions=EditCaptions(images),
        remove_images=RemoveImages(images),
        add_to_set=AddToSet(images),
        delete_set=DeleteSet(images),
        delete_cache=DeleteCache(images),
        start_import=StartImport(images),
        imports=ListImports(images),
    )


def _graphs(infra: Infrastructure, kit: GraphKit) -> GraphServices:
    executions, catalog = infra.executions, kit.catalog
    return GraphServices(
        node_catalog=GetNodeCatalog(catalog=catalog),
        resolve_node=ResolveNode(catalog=catalog),
        model_files=ListModelFiles(catalog=catalog),
        image_libraries=ListImageLibraries(catalog=catalog),
        list_starters=ListStarters(starters=NextStarters()),
        build_starter=BuildStarter(starters=NextStarters()),
        diagnostics=NodeDiagnostics(catalog=catalog),
        validate=ValidateGraph(runtime=kit.runtime),
        preview_memory=PreviewGraphMemory(memory_ledger=kit.memory_ledger, peak_source=kit.peak_source.observed),
        start_execution=kit.start_execution,
        list_executions=ListGraphExecutions(executions=executions),
        get_execution=GetGraphExecution(executions=executions),
        execution_metrics=GetExecutionMetrics(executions=executions, metrics=kit.metrics),
        rename_execution=RenameGraphExecution(executions=executions),
        send_action=SendExecutionAction(
            executions=executions, catalog=catalog, channel=FileCommandChannel(kit.scratch_dir)
        ),
        stop_execution=StopGraphExecution(
            executions=executions, writer=kit.writer, launcher=kit.supervisor, clock=infra.clock
        ),
        delete_executions=DeleteGraphExecutions(
            executions=executions,
            events=infra.publisher,
            scratch=kit.sweep_scratch,
            memory_ledger=kit.memory_ledger,
        ),
        reconcile_executions=ReconcileGraphExecutions(
            executions=executions,
            writer=kit.writer,
            launcher=kit.supervisor,
            clock=infra.clock,
            memory_ledger=kit.memory_ledger,
        ),
        save_graph=SaveGraph(library=kit.graph_library),
        get_graph=GetGraph(library=kit.graph_library),
        list_graphs=ListGraphs(library=kit.graph_library),
        delete_graph=DeleteGraph(library=kit.graph_library),
    )


def _apps(kit: GraphKit) -> AppServices:
    library = kit.app_library
    return AppServices(
        describe=kit.describe_modules,
        resolve=kit.resolve_board,
        save=SaveApp(library=library),
        get=GetApp(library=library),
        list=ListApps(library=library),
        delete=DeleteApp(library=library),
        diff=DiffApp(library=library, resolve=kit.resolve_board),
        run=RunApp(library=library, start=kit.start_execution),
    )
