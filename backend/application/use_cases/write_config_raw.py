"""WriteConfigRaw -- validate a full config document, then write it.

Create-or-replace (PUT semantics): the file and its directory appear
if needed. Validation precedes the write, so a rejected document
never truncates an existing file.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import InvalidQueryError
from ..ports.config_files import ConfigFiles


class WriteConfigRaw:
    def __init__(self, *, files: ConfigFiles, project_root: Path) -> None:
        self._files = files
        self._root = project_root

    def execute(self, path: str, content: str) -> None:
        if not path:
            raise InvalidQueryError("config path is required")
        if not isinstance(content, str):
            raise InvalidQueryError("content must be a string")
        self._files.replace(self._resolve(path), content)

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self._root / candidate
