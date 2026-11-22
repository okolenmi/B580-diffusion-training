"""Settings view and update."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ...application.ports.settings_store import SettingsChanges, SettingsView


class SettingsViewOut(BaseModel):
    """Raw stored values and what they currently resolve to (a
    ``null`` resolved entry means nothing can resolve it)."""

    stored: dict[str, str]
    resolved: dict[str, str | None]
    #: Whether the resolved interpreter file exists right now; a stored interpreter whose virtualenv was deleted is used as it is, not dropped.
    exists: dict[str, bool] = {}


class SettingsIn(BaseModel):
    """Partial update: absent key = untouched, empty string = clear
    the override and fall back to auto-detection."""

    comfy_dir: str | None = None
    venv_python: str | None = None
    checkpoints_dir: str | None = None
    loras_dir: str | None = None
    models_dir: str | None = None

    def to_changes(self) -> SettingsChanges:
        return SettingsChanges(
            comfy_dir=self.comfy_dir,
            venv_python=self.venv_python,
            checkpoints_dir=self.checkpoints_dir,
            loras_dir=self.loras_dir,
            models_dir=self.models_dir,
        )


def settings_out(view: SettingsView) -> SettingsViewOut:
    python = view.resolved.get("venv_python")
    # Folders that vanish resolve as unset, but an interpreter is used as stored (a stale one fails loudly at spawn), so only it can be
    # reported as missing. A bare command like "python" names no file to look for.
    exists = {"venv_python": Path(python).is_file()} if python and ("/" in python or "\\" in python) else {}
    return SettingsViewOut(stored=dict(view.stored), resolved=dict(view.resolved), exists=exists)


__all__ = ["SettingsViewOut", "SettingsIn", "settings_out"]
