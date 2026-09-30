"""ReadConfigRaw -- a config file's exact text (raw editor)."""

from __future__ import annotations

from pathlib import Path

from ..dto import RawConfig
from ..errors import InvalidQueryError
from ..ports.config_files import ConfigFiles


class ReadConfigRaw:
    def __init__(self, *, files: ConfigFiles, project_root: Path) -> None:
        self._files = files
        self._root = project_root

    def execute(self, path: str) -> RawConfig:
        if not path:
            raise InvalidQueryError("config path is required")
        return RawConfig(content=self._files.read_raw(self._resolve(path)))

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self._root / candidate
