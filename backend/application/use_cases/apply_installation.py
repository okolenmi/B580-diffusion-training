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


class ApplyInstallation:
    def __init__(self, settings: SettingsStore, project_root: Path) -> None:
        self._settings = settings
        self._project_root = project_root

    def execute(self) -> InstallationState:
        return self._state_of(self._settings.read())

    def apply(self, changes: SettingsChanges) -> SettingsView:
        """Validate and persist, but only while unconfigured. Returns the fresh view, as the settings API does."""
        state = self._state_of(self._settings.read())
        if state.configured:
            raise InstallerNotAllowedError(
                "this installation is already configured "
                f"(checkpoints in {state.resolved_checkpoints_dir}); change the folders from Settings instead")
        return self._settings.update(changes)

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
