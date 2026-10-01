"""SqliteSettingsStore -- settings persisted in the backend database.

Validation and persistence are separated so an update is atomic:
every provided value is checked first (``SettingsInvalidError`` with
the full per-key map rejects the request before anything is
written), then all changes commit in one transaction.

Validation is **pure**: it looks, it never acts. Checking a path by
creating it would let a rejected request leave a directory behind (the
exact shape docs 07 F-15 reproduced) and would let any unauthenticated
caller make this server build arbitrary directory trees (F-06). The
managed directories a valid update points at are created *after* the
commit, in :meth:`update`.

``get`` never raises -- path resolution calls it on every access and
a corrupted settings table must degrade to defaults, not take the
server down.
"""

from __future__ import annotations

import logging
import os
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

logger = logging.getLogger(__name__)


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
        self._create_managed_dirs(changes)
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
        """Check every provided value without touching the filesystem."""
        errors: dict[str, str] = {}

        if changes.comfy_dir and not Path(changes.comfy_dir).is_dir():
            errors["comfy_dir"] = f"'{changes.comfy_dir}' is not a directory"

        if changes.venv_python:
            interpreter = Path(changes.venv_python)
            if not interpreter.is_file():
                errors["venv_python"] = f"'{changes.venv_python}' is not a file"
            elif not os.access(interpreter, os.X_OK):
                # This value is *executed* on every start. "It exists" is
                # not a sufficient contract: a text file or a
                # non-executable script would fail later as an opaque
                # spawn error, so it is refused here, by name
                # (docs 07 F-06).
                errors["venv_python"] = (
                    f"'{changes.venv_python}' is not executable"
                )

        # checkpoints/loras are *managed* by this tool, so a missing
        # directory is acceptable -- but "acceptable" is decided by
        # looking at the parent, never by creating anything (F-15, F-06).
        for key in ("checkpoints_dir", "loras_dir"):
            value = getattr(changes, key)
            if value:
                errors.update(self._creatable_problem(key, value))

        return errors

    @staticmethod
    def _creatable_problem(key: str, value: str) -> dict[str, str]:
        """Can this directory exist? Decided by looking, never by trying.

        "Creatable" means the nearest ancestor that exists is a writable
        directory -- so a fresh ``~/models/loras/experiments`` is still
        accepted (the directory is created after the commit, as before),
        while a path under a file, or on a filesystem that would refuse,
        is reported instead of attempted.
        """
        path = Path(value)
        if not path.is_absolute():
            return {key: f"'{value}' must be an absolute path"}
        if path.is_dir():
            return {}
        if path.exists():
            return {key: f"'{value}' exists and is not a directory"}
        ancestor = path.parent
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            return {key: f"'{value}' cannot be created: '{ancestor}' is not a directory"}
        if not os.access(ancestor, os.W_OK | os.X_OK):
            return {key: f"'{value}' cannot be created: '{ancestor}' is not writable"}
        return {}

    @staticmethod
    def _create_managed_dirs(changes: SettingsChanges) -> None:
        """Create the managed directories a *committed* update points at.

        Deliberately after the commit: an update that fails validation
        leaves the disk exactly as it was (docs 07 F-15). A creation that
        still fails here (a race, a full disk) leaves the stored override
        inert -- the resolvers only accept an existing directory -- so it
        is logged rather than silently swallowed.
        """
        for key in ("checkpoints_dir", "loras_dir"):
            value = getattr(changes, key)
            if not value:
                continue
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning(
                    "%s is stored as '%s' but the directory could not be "
                    "created (it will resolve as unset until it exists): %s",
                    key, value, exc,
                )
