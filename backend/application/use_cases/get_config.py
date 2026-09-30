"""GetConfig -- read one config file as nested JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..errors import InvalidQueryError
from ..ports.config_files import ConfigFiles


class GetConfig:
    def __init__(self, *, files: ConfigFiles, project_root: Path) -> None:
        self._files = files
        self._root = project_root

    def execute(self, path: str) -> dict[str, Any]:
        """Return the validated config. Relative paths anchor at the
        project root; a missing/invalid file raises the config errors."""
        if not path:
            raise InvalidQueryError("config path is required")
        return self._files.read(self._resolve(path))

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self._root / candidate
