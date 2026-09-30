"""SqliteSettingsStore -- settings persisted in the backend database.

Validation and persistence are separated so an update is atomic:
every provided value is checked first (``SettingsInvalidError`` with
the full per-key map rejects the request before anything is
written), then all changes commit in one transaction.

``get`` never raises -- path resolution calls it on every access and
a corrupted settings table must degrade to defaults, not take the
server down.
"""

from __future__ import annotations

from pathlib import Path

from ..application.errors import SettingsInvalidError
from ..application.ports.settings_store import (
    SETTINGS_KEYS,
    SettingsChanges,
    SettingsStore,
    SettingsView,
)
from . import path_tiers
from .persistence.sqlite import SqliteDatabase


class SqliteSettingsStore(SettingsStore):
    def __init__(self, database: SqliteDatabase, project_root: Path) -> None:
        self._db = database
        self._root = project_root

    # -- SettingsStore -------------------------------------------------

    def read(self) -> SettingsView:
        stored = {key: self.get(key, "") for key in SETTINGS_KEYS}
        try:
            comfy = str(path_tiers.comfy_dir(self._root, self.get))
        except RuntimeError:
            comfy = None  # nothing identifies a ComfyUI install
        resolved = {
            "comfy_dir": comfy,
            "venv_python": path_tiers.venv_python(self._root, self.get),
            "checkpoints_dir": str(path_tiers.checkpoints_dir(self._root, self.get)),
            "loras_dir": str(path_tiers.loras_dir(self._root, self.get)),
        }
        return SettingsView(stored=stored, resolved=resolved)

    def update(self, changes: SettingsChanges) -> SettingsView:
        errors = self._validate(changes)
        if errors:
            raise SettingsInvalidError("invalid settings", details=errors)
        with self._db.connection() as conn:
            for key in SETTINGS_KEYS:
                value = getattr(changes, key)
                if value is None:
                    continue  # absent from this update: untouched
                if value == "":
                    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
                else:
                    conn.execute(
                        "INSERT INTO settings (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value),
                    )
        return self.read()

    def get(self, key: str, default: str = "") -> str:
        try:
            with self._db.connection() as conn:
                row = conn.execute(
                    "SELECT value FROM settings WHERE key = ?", (key,)
                ).fetchone()
            return str(row["value"]) if row is not None else default
        except Exception:  # noqa: BLE001 -- storage failure must not crash reads
            return default

    # -- validation ----------------------------------------------------

    def _validate(self, changes: SettingsChanges) -> dict[str, str]:
        errors: dict[str, str] = {}

        if changes.comfy_dir and not Path(changes.comfy_dir).is_dir():
            errors["comfy_dir"] = f"'{changes.comfy_dir}' is not a directory"

        if changes.venv_python and not Path(changes.venv_python).is_file():
            errors["venv_python"] = f"'{changes.venv_python}' is not a file"

        # checkpoints/loras are meant to be *managed* by this tool, so a
        # missing directory is created rather than rejected (unlike
        # comfy_dir/venv_python, which must point at existing things).
        for key in ("checkpoints_dir", "loras_dir"):
            value = getattr(changes, key)
            if not value:
                continue
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                errors[key] = f"could not create '{value}': {exc}"

        return errors
