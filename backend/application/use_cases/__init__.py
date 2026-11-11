"""Use cases -- one class per scenario, each with an ``execute``."""

from __future__ import annotations

from .apply_installation import (
    ApplyInstallation,
    InstallationState,
)
from .browse_assets import BrowseAssets
from .check_comfy_conflicts import (
    CheckComfyConflicts,
    ConflictReport,
)
from .install_packages import (
    GetInstall,
    InstallJob,
    InstallJobNotFound,
    JobNotFound,
    StartInstall,
)
from .check_requirements import (
    CheckRequirements,
    DescribeRequirements,
    ReadinessReport,
    RequirementStatus,
)
from .delete_graph import DeleteGraph
from .delete_graph_executions import DeleteGraphExecutions
from .get_config import GetConfig
from .get_config_options import GetConfigOptions
from .get_graph import GetGraph
from .get_execution_metrics import GetExecutionMetrics
from .get_graph_execution import GetGraphExecution
from .get_settings import GetSettings
from .inspect_asset import InspectAsset
from .list_assets import ListAssets
from .list_graph_executions import ListGraphExecutions
from .list_graphs import ListGraphs
from .make_asset_folder import MakeAssetFolder
from .node_diagnostics import NodeDiagnostics
from .node_shapes import GetNodeCatalog, ListImageLibraries, ListModelFiles, ResolveNode
from .read_config_raw import ReadConfigRaw
from .reconcile_graph_executions import ReconcileGraphExecutions
from .sweep_execution_scratch import (
    SweepExecutionScratch,
    SweepExecutionScratchResult,
)
from .save_graph import SaveGraph
from .start_graph_execution import StartGraphExecution
from .stop_graph_execution import StopGraphExecution
from .subscribe_monitor import SubscribeMonitor
from .update_config import UpdateConfig
from .update_settings import UpdateSettings
from .upload_asset import UploadAsset
from .preview_graph_memory import PreviewGraphMemory
from .validate_graph import ValidateGraph
from .write_config_raw import WriteConfigRaw

__all__ = [
    "ApplyInstallation",
    "BrowseAssets",
    "CheckComfyConflicts",
    "InstallJob",
    "StartInstall",
    "GetInstall",
    "JobNotFound",
    "InstallJobNotFound",
    "ConflictReport",
    "CheckRequirements",
    "DescribeRequirements",
    "InstallationState",
    "ReadinessReport",
    "RequirementStatus",
    "DeleteGraph",
    "DeleteGraphExecutions",
    "GetConfig",
    "GetConfigOptions",
    "GetGraph",
    "GetExecutionMetrics",
    "GetGraphExecution",
    "GetSettings",
    "InspectAsset",
    "ListAssets",
    "ListGraphExecutions",
    "ListGraphs",
    "MakeAssetFolder",
    "NodeDiagnostics",
    "GetNodeCatalog",
    "ResolveNode",
    "ListImageLibraries",
    "ListModelFiles",
    "ReadConfigRaw",
    "ReconcileGraphExecutions",
    "SaveGraph",
    "SweepExecutionScratch",
    "SweepExecutionScratchResult",
    "StartGraphExecution",
    "StopGraphExecution",
    "SubscribeMonitor",
    "UpdateConfig",
    "UpdateSettings",
    "UploadAsset",
    "PreviewGraphMemory",
    "ValidateGraph",
    "WriteConfigRaw",
]
