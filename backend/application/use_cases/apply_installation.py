"""ApplyInstallation -- write what the setup page collected, once.

**The gate.** An installer write is refused once the installation is configured (``installer_not_allowed``): the installer can
write filesystem paths and run installs, so a setup page left open in a tab on a working machine must not be able to re-point the
model folders under a run that is using them. Settings is where they change afterwards.

**Reopening.** A configured installation whose chosen environment is gone (the stored interpreter is not a file any more) opens the
installer again, because nothing can train without it and the setup page is the only place that rebuilds one. It stays open for the
rest of this server run, even once the rebuild has put the folder back: the page still has to store the choice and restart the server
after that, and the restart is what closes it again. A working environment is never offered for reinstalling; delete its folder to do that.

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
    #: The stored interpreter that no longer exists, while the installer is reopened for it.
    missing_environment: str | None = None

    @property
    def open(self) -> bool:
        """Whether the installer accepts writes."""
        return not self.configured or self.missing_environment is not None


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
        self._reopened_for: str | None = None

    def execute(self) -> InstallationState:
        return self._state_of(self._settings.read())

    def apply(self, changes: SettingsChanges, *, restart: bool = False) -> ApplyResult:
        """Validate and persist, but only while unconfigured.

        The server restarts into the chosen environment when that is not the one it runs in, and always when ``restart`` says a
        package was just installed: a running process keeps what it imported (and what it failed to import) before the install, even
        when the environment is the same one.
        """
        self.require_open()
        view = self._settings.update(changes)
        interpreter = self._chosen(view) if changes.venv_python else None
        return ApplyResult(view, *self._restart(interpreter, force=restart))

    def server_runs_in(self, interpreter: str | None) -> bool | None:
        """Whether this server is running in ``interpreter``'s environment; None when that cannot be told."""
        if not interpreter or self._restarter is None:
            return None
        return self._restarter.running_in(interpreter)

    def restart(self) -> ApplyResult:
        """Restart now into the stored environment: the page's own button, for a restart that did not happen by itself."""
        self.require_open()
        view = self._settings.read()
        return ApplyResult(view, *self._restart(self._chosen(view), force=True))

    @staticmethod
    def _chosen(view: SettingsView) -> str | None:
        """The environment the person picked, which is the *stored* value.

        Not ``resolved``: that puts the ``VENV_PYTHON`` variable first, and a server started through the bootstrap has that variable
        pointing at its own temporary environment, so restarting "into" the resolved one re-launched the same temporary environment.
        """
        return view.stored.get("venv_python") or None

    def require_open(self) -> None:
        state = self._state_of(self._settings.read())
        if not state.open:
            raise InstallerNotAllowedError(
                "this installation is already configured "
                f"(checkpoints in {state.resolved_checkpoints_dir}); change the folders from Settings instead")

    def _restart(self, interpreter: str | None, *, force: bool) -> tuple[bool, str | None]:
        restarter = self._restarter
        if not interpreter or restarter is None:
            return False, ("there is no environment to restart into yet" if force else None)
        if restarter.running_in(interpreter) and not force:
            return False, None
        if not restarter.can_serve(interpreter):
            return False, (f"{interpreter} cannot run the server (it needs Python 3.14 or newer and the server's own packages), so the "
                           "server stays in its temporary environment and most nodes will not load. Choose another environment.")
        restarter.restart_into(interpreter)
        return True, None

    def _state_of(self, view: SettingsView) -> InstallationState:
        resolved = view.resolved
        stored = view.stored.get("venv_python")
        if stored and not Path(stored).is_file():
            self._reopened_for = stored
        missing = tuple(key for key in ("checkpoints_dir", "loras_dir") if not resolved.get(key))
        return InstallationState(
            configured=not missing,
            resolved_checkpoints_dir=resolved.get("checkpoints_dir"),
            resolved_loras_dir=resolved.get("loras_dir"),
            resolved_models_dir=resolved.get("models_dir"),
            default_models_dir=str(self._project_root / "models"),
            missing=missing,
            missing_environment=self._reopened_for,
        )
