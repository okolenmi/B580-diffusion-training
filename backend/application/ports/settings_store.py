"""SettingsStore port -- persisted server-level settings.

Two views over one small key/value table:

* ``stored``  -- exactly what is persisted per key (``""`` = unset);
* ``resolved`` -- what each path setting currently points at after
  applying the full resolution policy (environment, workspace
  conventions, then the stored override), or ``None`` when nothing
  can resolve it (only ``comfy_dir`` can fail).

Updates are atomic: every provided value is validated first, then
all of them persist together -- a rejected request changes nothing
(``SettingsInvalidError`` carries a per-key message map). Absent keys
in a change set are left alone; an explicit empty string clears the
override.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

SETTINGS_KEYS: tuple[str, ...] = (
    "default_config",
    "comfy_dir",
    "venv_python",
    "checkpoints_dir",
    "loras_dir",
)

RESOLVED_KEYS: tuple[str, ...] = (
    "comfy_dir",
    "venv_python",
    "checkpoints_dir",
    "loras_dir",
)


@dataclass(frozen=True, slots=True)
class SettingsView:
    """Snapshot: raw stored values + what they resolve to."""

    stored: dict[str, str]
    resolved: dict[str, str | None]


@dataclass(frozen=True, slots=True)
class SettingsChanges:
    """Partial update. ``None`` = leave untouched; ``""`` = clear."""

    default_config: str | None = None
    comfy_dir: str | None = None
    venv_python: str | None = None
    checkpoints_dir: str | None = None
    loras_dir: str | None = None


class SettingsStore(ABC):
    @abstractmethod
    def read(self) -> SettingsView:
        raise NotImplementedError

    @abstractmethod
    def update(self, changes: SettingsChanges) -> SettingsView:
        raise NotImplementedError

    @abstractmethod
    def get(self, key: str, default: str = "") -> str:
        """Raw stored value; never raises (path resolution must not
        be crashable by a broken database)."""
        raise NotImplementedError
