"""Environment inspection and the installer wizard."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from ...application.ports.settings_store import SettingsChanges
from ...application.use_cases.apply_installation import InstallationState
from ...application.ports.requirements_manifest import TORCH_TESTED_VERSION
from ...application.use_cases.inspect_environment import EnvironmentReport


class InstallerStateOut(BaseModel):
    configured: bool
    #: Whether the installer accepts writes: unconfigured, or reopened because the chosen environment is gone.
    setup_open: bool
    #: The environment chosen earlier that no longer exists, while setup is reopened for it.
    missing_environment: str | None = None
    resolved_checkpoints_dir: str | None
    resolved_loras_dir: str | None
    resolved_models_dir: str | None
    #: Where the model folder goes when the user accepts the default.
    default_models_dir: str
    missing: list[str]
    #: The latest install job, running or finished, so a reloaded setup page can resume it.
    install_id: str | None = None


class InstallerDeviceOut(BaseModel):
    """One card. ``index`` is torch's device index, not the position in the list."""

    index: int
    present: bool
    name: str | None
    total_memory_mb: float | None
    reason: str | None
    #: Set when the row did not come from torch (see ``SystemCards``).
    note: str | None = None


class InstallerDevicesOut(BaseModel):
    """Every card, and whether anybody looked: an empty list from a probe that ran differs from one that failed."""

    enumerated: bool
    backend: str
    devices: list[InstallerDeviceOut]
    reason: str | None


class PackageStateOut(BaseModel):
    name: str
    tier: str
    why: str
    approx_mb: int | None
    installed_version: str | None


class InstallerEnvironmentOut(BaseModel):
    """What an environment holds, and what installing into it would do."""

    kind: str
    venv_dir: str
    interpreter: str | None
    python_version: str | None
    error: str | None
    packages: list[PackageStateOut]
    to_install: list[str]
    download_mb: int
    installed_count: int
    problems: list[str]
    #: What the install deletes first: only ever the project's own virtualenv.
    discards: list[str] = []
    usable: bool
    #: The torch release offered as the default.
    tested_torch: str


class InstallerVerifyOut(BaseModel):
    present: bool
    name: str | None
    total_memory_mb: int | None
    torch_version: str | None
    reason: str | None
    #: Whether the server itself runs in that environment; False means its nodes cannot import what was installed there.
    server_in_environment: bool | None = None


class InstallerInstallIn(BaseModel):
    """Where to install. What gets installed is decided by the server from the environment itself."""

    target: Literal["new", "existing"]
    #: The virtualenv folder; required for ``existing``.
    venv_dir: str | None = None
    #: Which torch to install when the environment has none: ``tested`` (the release the trainer is tested with), ``latest``
    #: (what the Intel XPU index offers) or an exact version such as ``2.11.0``.
    torch: str = "tested"


class InstallerInstallOut(BaseModel):
    """The job. ``log`` is the tail of pip's output and ``target_python`` is set once it succeeded."""

    id: str
    state: str
    target_label: str
    packages: list[str]
    log: list[str]
    error: str | None
    target_python: str | None
    terminal: bool


class InstallerApplyIn(BaseModel):
    """What the wizard collected; an absent key is left alone."""

    models_dir: str | None = None
    venv_python: str | None = None
    #: A package was just installed: restart the server even if it already runs in that environment.
    restart: bool = False

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(models_dir=self.models_dir, venv_python=self.venv_python)


class InstallerApplyOut(BaseModel):
    """The fresh settings view plus the new state, so the client learns whether setup is finished from the server's own read."""

    settings: dict[str, Any]
    state: dict[str, Any]
    #: True when the server is restarting into the chosen environment; the page waits for /health before leaving.
    restarting: bool = False
    #: Set when the server could not move into the chosen environment, and why.
    restart_note: str | None = None


def environment_out(report: EnvironmentReport) -> InstallerEnvironmentOut:
    return InstallerEnvironmentOut(
        kind=report.kind,
        venv_dir=report.venv_dir,
        interpreter=report.interpreter,
        python_version=report.python_version,
        error=report.error,
        packages=[PackageStateOut(name=p.name, tier=p.tier, why=p.why, approx_mb=p.approx_mb, installed_version=p.installed_version)
                  for p in report.packages],
        to_install=list(report.to_install),
        download_mb=report.download_mb,
        installed_count=report.installed_count,
        problems=list(report.problems),
        discards=list(report.discards),
        usable=report.usable,
        tested_torch=TORCH_TESTED_VERSION,
    )


def state_out(state: InstallationState, install: dict | None = None) -> InstallerStateOut:
    return InstallerStateOut(
        install_id=install["id"] if install else None,
        configured=state.configured,
        setup_open=state.open,
        missing_environment=state.missing_environment,
        resolved_checkpoints_dir=state.resolved_checkpoints_dir,
        resolved_loras_dir=state.resolved_loras_dir,
        resolved_models_dir=state.resolved_models_dir,
        default_models_dir=state.default_models_dir,
        missing=list(state.missing),
    )


__all__ = ["InstallerStateOut", "InstallerDeviceOut", "InstallerDevicesOut", "PackageStateOut", "InstallerEnvironmentOut", "InstallerVerifyOut", "InstallerInstallIn", "InstallerInstallOut", "InstallerApplyIn", "InstallerApplyOut", "environment_out", "state_out"]
