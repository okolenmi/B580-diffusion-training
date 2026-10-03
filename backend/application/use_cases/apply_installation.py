"""ApplyInstallation -- write the paths a wizard collected, once.

The design's §5 plus the tail of the flow: after the user has answered
"where do model files live", the server needs to *apply* that answer, and
the interesting part is not the writing -- `SqliteSettingsStore` already
does that atomically -- it is refusing to do it at the wrong time and
refusing to do it halfway.

**The gate.** An installer write is refused once this installation is
configured. That is the `installer_not_allowed` error, and it exists
because the installer is the one surface in this project with a real
capability: it writes filesystem paths and, in the phase after this,
would run package installs. Keeping it reachable only in the unconfigured
state means a wizard left open in a tab on a working machine cannot
re-point the model directories out from under a run that is using them.
So "is this installation configured?" is a real predicate and it lives
here, not in a route.

**Atomicity comes from the store, not from here.** Every path is validated
before any is written; a rejected pair changes nothing. What this adds is
the ordering that a store cannot know about: *check the gate before
validating*, so a refused request does not report a validation error the
user has to interpret as the real problem.

**What "configured" means.** `comfy_dir` and `checkpoints_dir` both
resolve and `loras_dir` resolves. Deliberately not the device and not the
training stack: those are the installer's job to *report*, and a machine
with no GPU still has chosen paths. Gating on them would make the wizard
impossible to finish on a machine where the card is missing -- which is
exactly when someone needs it most.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..errors import InstallerNotAllowedError
from ..ports.settings_store import SettingsChanges, SettingsStore, SettingsView


@dataclass(frozen=True, slots=True)
class InstallationState:
    """Whether the wizard still has anything to do.

    `missing` names the specific settings, so the wizard can render "two
    things left" rather than a yes/no that gives the user nothing to act
    on.
    """

    configured: bool
    resolved_comfy_dir: str | None
    resolved_checkpoints_dir: str | None
    resolved_loras_dir: str | None
    missing: tuple[str, ...]


class ApplyInstallation:
    def __init__(self, settings: SettingsStore) -> None:
        self._settings = settings

    def execute(self) -> InstallationState:
        view = self._settings.read()
        return self._state_of(view)

    def apply(self, changes: SettingsChanges) -> SettingsView:
        """Validate and persist, but only while unconfigured.

        Returns the fresh view -- the same convention as the settings API,
        so a client never has to re-GET to learn what took effect.
        """
        state = self._state_of(self._settings.read())
        if state.configured:
            raise InstallerNotAllowedError(
                "this installation is already configured "
                f"(comfy_dir={state.resolved_comfy_dir}, "
                f"checkpoints_dir={state.resolved_checkpoints_dir}); "
                "change these from Settings instead"
            )
        return self._settings.update(changes)

    # -- the predicate --------------------------------------------------

    def _state_of(self, view: SettingsView) -> InstallationState:
        resolved = view.resolved
        missing = tuple(
            key for key in ("comfy_dir", "checkpoints_dir", "loras_dir")
            if not resolved.get(key)
        )
        return InstallationState(
            configured=not missing,
            resolved_comfy_dir=resolved.get("comfy_dir"),
            resolved_checkpoints_dir=resolved.get("checkpoints_dir"),
            resolved_loras_dir=resolved.get("loras_dir"),
            missing=missing,
        )


def default_checkpoints_dir(comfy_dir: str | Path) -> Path:
    """Where a checkpoint goes when the user has not chosen otherwise.

    ComfyUI's own `models/checkpoints`, because that is what
    `paths.get_checkpoints_dir()` resolves and choosing anything else
    would mean the wizard's default disagreed with the resolution policy
    -- which is the kind of disagreement that surfaces later as "I set it
    to the default and it went somewhere else".
    """
    return Path(comfy_dir) / "models" / "checkpoints"


def default_loras_dir(comfy_dir: str | Path) -> Path:
    return Path(comfy_dir) / "models" / "loras"