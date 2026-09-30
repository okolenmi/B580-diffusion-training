"""ApplicationServices -- the wired-up use cases plus shared ports.

The composition root (``backend.bootstrap``) fills this frozen
aggregate and hands it to presentation; it is the *only* object the
web layer may reach for. Holding it in the application layer (rather
than in presentation) keeps the dependency direction honest: the web
layer depends on application, never the other way around.

Use cases are grouped by domain so the aggregate stays navigable as
it grows: ``runs`` (the original flat fields, kept flat for
continuity), ``config``, ``settings``, ``assets``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ports.event_bus import EventBus
from .use_cases.browse_assets import BrowseAssets
from .use_cases.delete_runs import DeleteRuns
from .use_cases.get_active_run import GetActiveRun
from .use_cases.get_config import GetConfig
from .use_cases.get_config_options import GetConfigOptions
from .use_cases.get_run import GetRun
from .use_cases.get_run_log import GetRunLog
from .use_cases.get_settings import GetSettings
from .use_cases.get_start_options import GetStartOptions
from .use_cases.inspect_asset import InspectAsset
from .use_cases.list_assets import ListAssets
from .use_cases.list_runs import ListRuns
from .use_cases.make_asset_folder import MakeAssetFolder
from .use_cases.read_config_raw import ReadConfigRaw
from .use_cases.reconcile_runs import ReconcileRuns
from .use_cases.start_training import StartTraining
from .use_cases.stop_training import StopTraining
from .use_cases.update_config import UpdateConfig
from .use_cases.update_settings import UpdateSettings
from .use_cases.upload_asset import UploadAsset
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
    # shared
    event_bus: EventBus
