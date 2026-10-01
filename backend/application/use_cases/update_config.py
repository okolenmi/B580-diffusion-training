"""UpdateConfig -- merge a partial config into a config file.

The config file must already exist (this is a merge, not a create);
``overrides`` is a partial nested dict matching the config's
structure. Validation happens before any write: a rejected update
leaves the file untouched. Nothing here knows about launch state --
saving never clears, migrates, or reinterprets fields.
"""

from __future__ import annotations

from typing import Any

from ..errors import InvalidQueryError
from ..ports.config_files import ConfigFiles
from ..project_paths import ProjectPaths


class UpdateConfig:
    def __init__(self, *, files: ConfigFiles, paths: ProjectPaths) -> None:
        self._files = files
        self._paths = paths

    def execute(self, path: str, overrides: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(overrides, dict):
            raise InvalidQueryError("overrides must be an object")
        return self._files.update(self._paths.config(path), overrides)
