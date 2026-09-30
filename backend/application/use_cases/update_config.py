"""UpdateConfig -- merge a partial config into a config file.

The config file must already exist (this is a merge, not a create);
``overrides`` is a partial nested dict matching the config's
structure. Validation happens before any write: a rejected update
leaves the file untouched. Nothing here knows about launch state --
saving never clears, migrates, or reinterprets fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import InvalidQueryError
from ..ports.config_files import ConfigFiles


class UpdateConfig:
    def __init__(self, *, files: ConfigFiles, project_root: Path) -> None:
        self._files = files
        self._root = project_root

    def execute(self, path: str, overrides: dict[str, Any]) -> dict[str, Any]:
        if not path:
            raise InvalidQueryError("config path is required")
        if not isinstance(overrides, dict):
            raise InvalidQueryError("overrides must be an object")
        return self._files.update(self._resolve(path), overrides)

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self._root / candidate
