"""ReadConfigRaw -- a config file's exact text (raw editor)."""

from __future__ import annotations


from ..dto import RawConfig
from ..ports.config_files import ConfigFiles
from ..project_paths import ProjectPaths


class ReadConfigRaw:
    def __init__(self, *, files: ConfigFiles, paths: ProjectPaths) -> None:
        self._files = files
        self._paths = paths

    def execute(self, path: str) -> RawConfig:
        return RawConfig(content=self._files.read_raw(self._paths.config(path)))
