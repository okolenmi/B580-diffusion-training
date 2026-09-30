"""UpdateSettings -- validate every provided value, then persist all.

Atomic by contract: if any value is rejected,
``SettingsInvalidError`` carries the full ``{key: message}`` map and
nothing is written. On success the caller gets the fresh view
(re-read after persisting), so a response is always authoritative.
"""

from __future__ import annotations

from ..ports.settings_store import SettingsChanges, SettingsStore, SettingsView


class UpdateSettings:
    def __init__(self, *, settings: SettingsStore) -> None:
        self._settings = settings

    def execute(self, changes: SettingsChanges) -> SettingsView:
        return self._settings.update(changes)
