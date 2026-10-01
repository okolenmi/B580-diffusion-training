"""Use cases -- one class per scenario, each with an ``execute``."""

from __future__ import annotations

from .browse_assets import BrowseAssets
from .bulk_update_dataset_items import BulkUpdateDatasetItems
from .commit_dataset_items import CommitDatasetItems
from .create_dataset import CreateDataset
from .delete_dataset import DeleteDataset
from .delete_graph import DeleteGraph
from .delete_graph_executions import DeleteGraphExecutions
from .delete_runs import DeleteRuns
from .discard_dataset_items import DiscardDatasetItems
from .get_active_run import GetActiveRun
from .get_config import GetConfig
from .get_config_options import GetConfigOptions
from .get_dataset import GetDataset
from .get_graph import GetGraph
from .get_graph_execution import GetGraphExecution
from .get_run import GetRun
from .get_run_log import GetRunLog
from .get_settings import GetSettings
from .get_start_options import GetStartOptions
from .inspect_asset import InspectAsset
from .list_assets import ListAssets
from .list_dataset_items import ListDatasetItems
from .list_dataset_sets import ListDatasetSets
from .list_dataset_tasks import ListDatasetTasks
from .list_datasets import ListDatasets
from .list_graph_executions import ListGraphExecutions
from .list_graphs import ListGraphs
from .list_node_catalog import ListNodeCatalog
from .list_runs import ListRuns
from .make_asset_folder import MakeAssetFolder
from .node_diagnostics import NodeDiagnostics
from .read_config_raw import ReadConfigRaw
from .read_dataset_file import ReadDatasetFile
from .reconcile_dataset_tasks import ReconcileDatasetTasks
from .reconcile_graph_executions import ReconcileGraphExecutions
from .reconcile_runs import ReconcileRuns
from .save_graph import SaveGraph
from .set_dataset_preview import SetDatasetPreview
from .start_dataset_task import StartDatasetTask
from .start_graph_execution import StartGraphExecution
from .start_training import StartTraining
from .stop_dataset_task import StopDatasetTask
from .stop_graph_execution import StopGraphExecution
from .stop_training import StopTraining
from .subscribe_monitor import SubscribeMonitor
from .update_config import UpdateConfig
from .update_dataset_item import UpdateDatasetItem
from .update_settings import UpdateSettings
from .upload_asset import UploadAsset
from .validate_graph import ValidateGraph
from .write_config_raw import WriteConfigRaw

__all__ = [
    "BrowseAssets",
    "BulkUpdateDatasetItems",
    "CommitDatasetItems",
    "CreateDataset",
    "DeleteDataset",
    "DeleteGraph",
    "DeleteGraphExecutions",
    "DeleteRuns",
    "DiscardDatasetItems",
    "GetActiveRun",
    "GetConfig",
    "GetConfigOptions",
    "GetDataset",
    "GetGraph",
    "GetGraphExecution",
    "GetRun",
    "GetRunLog",
    "GetSettings",
    "GetStartOptions",
    "InspectAsset",
    "ListAssets",
    "ListDatasetItems",
    "ListDatasetSets",
    "ListDatasetTasks",
    "ListDatasets",
    "ListGraphExecutions",
    "ListGraphs",
    "ListNodeCatalog",
    "ListRuns",
    "MakeAssetFolder",
    "NodeDiagnostics",
    "ReadConfigRaw",
    "ReadDatasetFile",
    "ReconcileDatasetTasks",
    "ReconcileGraphExecutions",
    "ReconcileRuns",
    "SaveGraph",
    "SetDatasetPreview",
    "StartDatasetTask",
    "StartGraphExecution",
    "StartTraining",
    "StopDatasetTask",
    "StopGraphExecution",
    "StopTraining",
    "SubscribeMonitor",
    "UpdateConfig",
    "UpdateDatasetItem",
    "UpdateSettings",
    "UploadAsset",
    "ValidateGraph",
    "WriteConfigRaw",
]
