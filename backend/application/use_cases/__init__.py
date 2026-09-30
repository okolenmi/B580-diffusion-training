"""Use cases -- one class per scenario, each with an ``execute``."""

from __future__ import annotations

from .browse_assets import BrowseAssets
from .delete_runs import DeleteRuns
from .get_active_run import GetActiveRun
from .get_config import GetConfig
from .get_config_options import GetConfigOptions
from .get_run import GetRun
from .get_run_log import GetRunLog
from .get_settings import GetSettings
from .get_start_options import GetStartOptions
from .inspect_asset import InspectAsset
from .list_assets import ListAssets
from .list_runs import ListRuns
from .make_asset_folder import MakeAssetFolder
from .read_config_raw import ReadConfigRaw
from .reconcile_runs import ReconcileRuns
from .start_training import StartTraining
from .stop_training import StopTraining
from .update_config import UpdateConfig
from .update_settings import UpdateSettings
from .upload_asset import UploadAsset
from .write_config_raw import WriteConfigRaw

__all__ = [
    "BrowseAssets",
    "DeleteRuns",
    "GetActiveRun",
    "GetConfig",
    "GetConfigOptions",
    "GetRun",
    "GetRunLog",
    "GetSettings",
    "GetStartOptions",
    "InspectAsset",
    "ListAssets",
    "ListRuns",
    "MakeAssetFolder",
    "ReadConfigRaw",
    "ReconcileRuns",
    "StartTraining",
    "StopTraining",
    "UpdateConfig",
    "UpdateSettings",
    "UploadAsset",
    "WriteConfigRaw",
]
