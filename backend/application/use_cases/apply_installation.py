"""ApplyInstallation -- write what the setup page collected, once.

**The gate.** An installer write is refused once the installation is configured (``installer_not_allowed``): the installer can
write filesystem paths and run installs, so a setup page left open in a tab on a working machine must not be able to re-point the
model folders under a run that is using them. Settings is where they change afterwards.

**Configured** means the checkpoint and LoRA folders resolve -- from the models folder the user chose, from an explicit setting, or
from a ComfyUI checkout found next to the project. Nothing else is required: ComfyUI is optional, and the graphics card and the
training packages are things the page reports, not conditions of finishing setup.

Every value is validated before any is written, so a rejected request changes nothing (the store guarantees that).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import InstallerNotAllowedError
from ..ports.server_restart import ServerRestarter
from ..ports.settings_store import SettingsChanges, SettingsStore, SettingsView


@dataclass(frozen=True, slots=True)
class InstallationState:
    """Whether setup still has anything to do; ``missing`` names the settings so the page can say what is left."""

    configured: bool
    resolved_checkpoints_dir: str | None
    resolved_loras_dir: str | None
    resolved_models_dir: str | None
    default_models_dir: str
    missing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ApplyResult:
    view: SettingsView
    #: The server is about to restart into the chosen environment; the page waits for it to come back.
    restarting: bool
    #: Why the server stays where it is although the chosen environment is a different one; None when nothing needs saying.
    restart_note: str | None = None


class ApplyInstallation:
    def __init__(self, settings: SettingsStore, project_root: Path, restarter: ServerRestarter | None = None) -> None:
        self._settings = settings
        self._project_root = project_root
        self._restarter = restarter

    def execute(self) -> InstallationState:
        return self._state_of(self._settings.read())

    def apply(self, changes: SettingsChanges) -> ApplyResult:
        """Validate and persist, but only while unconfigured.

        When the chosen environment is not the one this server runs in, the server restarts into it: the node library needs the
        packages that were just installed there.
        """
        state = self._state_of(self._settings.read())
        if state.configured:
            raise InstallerNotAllowedError(
                "this installation is already configured "
                f"(checkpoints in {state.resolved_checkpoints_dir}); change the folders from Settings instead")
        view = self._settings.update(changes)
        restarting, note = self._restart_if_needed(changes, view)
        return ApplyResult(view, restarting, note)

    def _restart_if_needed(self, changes: SettingsChanges, view: SettingsView) -> tuple[bool, str | None]:
        interpreter = view.resolved.get("venv_python") if changes.venv_python else None
        restarter = self._restarter
        if not interpreter or restarter is None or restarter.running_in(interpreter):
            return False, None
        if not restarter.can_serve(interpreter):
            return False, (f"{interpreter} cannot run the server (it needs Python 3.14 or newer and the server's own packages), so the "
                           "server stays in its temporary environment and most nodes will not load. Choose another environment.")
        restarter.restart_into(interpreter)
        return True, None

    def _state_of(self, view: SettingsView) -> InstallationState:
        resolved = view.resolved
        missing = tuple(key for key in ("checkpoints_dir", "loras_dir") if not resolved.get(key))
        return InstallationState(
            configured=not missing,
            resolved_checkpoints_dir=resolved.get("checkpoints_dir"),
            resolved_loras_dir=resolved.get("loras_dir"),
            resolved_models_dir=resolved.get("models_dir"),
            default_models_dir=str(self._project_root / "models"),
            missing=missing,
        )
