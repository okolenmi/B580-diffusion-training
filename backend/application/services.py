"""ApplicationServices -- the wired-up use cases plus shared ports.

The composition root (``backend.bootstrap``) fills this frozen
aggregate and hands it to presentation; it is the *only* object the
web layer may reach for. Holding it in the application layer (rather
than in presentation) keeps the dependency direction honest: the web
layer depends on application, never the other way around.

Use cases are grouped by domain so the aggregate stays navigable as
it grows: ``runs`` (the original flat fields, kept flat for
continuity), ``config``, ``settings``, ``assets``, ``datasets``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ports.event_bus import EventBus
from .use_cases.browse_assets import BrowseAssets
from .use_cases.bulk_update_dataset_items import BulkUpdateDatasetItems
from .use_cases.commit_dataset_items import CommitDatasetItems
from .use_cases.create_dataset import CreateDataset
from .use_cases.delete_dataset import DeleteDataset
from .use_cases.delete_graph import DeleteGraph
from .use_cases.delete_graph_executions import DeleteGraphExecutions
from .use_cases.delete_runs import DeleteRuns
from .use_cases.discard_dataset_items import DiscardDatasetItems
from .use_cases.get_active_run import GetActiveRun
from .use_cases.get_config import GetConfig
from .use_cases.get_config_options import GetConfigOptions
from .use_cases.get_dataset import GetDataset
from .use_cases.get_graph import GetGraph
from .use_cases.get_graph_execution import GetGraphExecution
from .use_cases.get_run import GetRun
from .use_cases.get_run_log import GetRunLog
from .use_cases.get_settings import GetSettings
from .use_cases.get_start_options import GetStartOptions
from .use_cases.inspect_asset import InspectAsset
from .use_cases.list_assets import ListAssets
from .use_cases.list_dataset_items import ListDatasetItems
from .use_cases.list_dataset_sets import ListDatasetSets
from .use_cases.list_dataset_tasks import ListDatasetTasks
from .use_cases.list_datasets import ListDatasets
from .use_cases.list_graph_executions import ListGraphExecutions
from .use_cases.list_graphs import ListGraphs
from .use_cases.list_node_catalog import ListNodeCatalog
from .use_cases.list_runs import ListRuns
from .use_cases.make_asset_folder import MakeAssetFolder
from .use_cases.node_diagnostics import NodeDiagnostics
from .use_cases.read_config_raw import ReadConfigRaw
from .use_cases.read_dataset_file import ReadDatasetFile
from .use_cases.reconcile_dataset_tasks import ReconcileDatasetTasks
from .use_cases.reconcile_graph_executions import ReconcileGraphExecutions
from .use_cases.subscribe_monitor import SubscribeMonitor
from .use_cases.reconcile_runs import ReconcileRuns
from .use_cases.save_graph import SaveGraph
from .use_cases.set_dataset_preview import SetDatasetPreview
from .use_cases.start_dataset_task import StartDatasetTask
from .use_cases.start_graph_execution import StartGraphExecution
from .use_cases.start_training import StartTraining
from .use_cases.stop_dataset_task import StopDatasetTask
from .use_cases.stop_graph_execution import StopGraphExecution
from .use_cases.stop_training import StopTraining
from .use_cases.update_config import UpdateConfig
from .use_cases.update_dataset_item import UpdateDatasetItem
from .use_cases.update_settings import UpdateSettings
from .use_cases.upload_asset import UploadAsset
from .use_cases.validate_graph import ValidateGraph
from .use_cases.write_config_raw import WriteConfigRaw


@dataclass(frozen=True, slots=True)
class ConfigServices:
    """Config-file document + schema + launch-picker operations."""

    read: GetConfig
    update: UpdateConfig
    read_raw: ReadConfigRaw
    write_raw: WriteConfigRaw
    options: GetConfigOptions
    start_options: GetStartOptions


@dataclass(frozen=True, slots=True)
class SettingsServices:
    read: GetSettings
    update: UpdateSettings


@dataclass(frozen=True, slots=True)
class AssetServices:
    list: ListAssets
    browse: BrowseAssets
    make_folder: MakeAssetFolder
    upload: UploadAsset
    inspect: InspectAsset


@dataclass(frozen=True, slots=True)
class DatasetServices:
    """Dataset library, curation, training sets, and task lifecycle (M3b)."""

    list: ListDatasets
    get: GetDataset
    create: CreateDataset
    delete: DeleteDataset
    items: ListDatasetItems
    update_item: UpdateDatasetItem
    bulk_update: BulkUpdateDatasetItems
    discard: DiscardDatasetItems
    sets: ListDatasetSets
    commit: CommitDatasetItems
    tasks: ListDatasetTasks
    start_task: StartDatasetTask
    stop_task: StopDatasetTask
    reconcile_tasks: ReconcileDatasetTasks
    read_file: ReadDatasetFile
    set_preview: SetDatasetPreview


@dataclass(frozen=True, slots=True)
class GraphServices:
    """Node-graph catalog, validation, execution, history, library (M4)."""

    catalog: ListNodeCatalog
    diagnostics: NodeDiagnostics
    validate: ValidateGraph
    start_execution: StartGraphExecution
    list_executions: ListGraphExecutions
    get_execution: GetGraphExecution
    stop_execution: StopGraphExecution
    delete_executions: DeleteGraphExecutions
    reconcile_executions: ReconcileGraphExecutions
    save_graph: SaveGraph
    get_graph: GetGraph
    list_graphs: ListGraphs
    delete_graph: DeleteGraph


@dataclass(frozen=True, slots=True)
class MonitorServices:
    """Live node telemetry (the monitor page's data path, M6).

    A group of its own rather than a raw port on the aggregate: the
    endpoint used to reach for ``services.monitor_bus`` directly, which
    left the one member of the aggregate that was *not* behind a use case
    (docs 08 S-04).
    """

    subscribe: SubscribeMonitor


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    # runs domain (M1/M2)
    list_runs: ListRuns
    get_run: GetRun
    delete_runs: DeleteRuns
    get_active_run: GetActiveRun
    start_training: StartTraining
    stop_training: StopTraining
    get_run_log: GetRunLog
    reconcile_runs: ReconcileRuns
    # config / settings / assets domains (M3)
    config: ConfigServices
    settings: SettingsServices
    assets: AssetServices
    # dataset domain (M3b)
    datasets: DatasetServices
    # graph domain (M4)
    graphs: GraphServices
    # monitor telemetry (M6)
    monitor: MonitorServices
    # shared
    event_bus: EventBus
