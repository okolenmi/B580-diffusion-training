"""GetConfig -- read one config file as nested JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ports.config_files import ConfigFiles
from ..project_paths import ProjectPaths


class GetConfig:
    def __init__(self, *, files: ConfigFiles, paths: ProjectPaths) -> None:
        self._files = files
        self._paths = paths

    def execute(self, path: str) -> dict[str, Any]:
        """Return the validated config. Relative paths anchor at the
        project root; a missing/invalid file raises the config errors."""
        return self._files.read(self._paths.config(path))
