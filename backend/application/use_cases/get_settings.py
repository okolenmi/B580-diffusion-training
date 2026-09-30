"""GetSettings -- stored values + what they currently resolve to."""

from __future__ import annotations

from ..ports.settings_store import SettingsStore, SettingsView


class GetSettings:
    def __init__(self, *, settings: SettingsStore) -> None:
        self._settings = settings

    def execute(self) -> SettingsView:
        return self._settings.read()
