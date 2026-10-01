"""WriteConfigRaw -- validate a full config document, then write it.

Create-or-replace (PUT semantics): the file and its directory appear
if needed. Validation precedes the write, so a rejected document
never truncates an existing file.
"""

from __future__ import annotations


from ..errors import InvalidQueryError
from ..ports.config_files import ConfigFiles
from ..project_paths import ProjectPaths


class WriteConfigRaw:
    def __init__(self, *, files: ConfigFiles, paths: ProjectPaths) -> None:
        self._files = files
        self._paths = paths

    def execute(self, path: str, content: str) -> None:
        if not isinstance(content, str):
            raise InvalidQueryError("content must be a string")
        self._files.replace(self._paths.config(path), content)
