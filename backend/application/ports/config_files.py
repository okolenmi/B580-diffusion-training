"""ConfigFiles port -- document operations on training config files.

The config file is the single source of truth for a training
configuration; this port reads it, merges explicit partial updates
into it, and replaces its raw text. Nothing else in the backend
writes it, and no other operation (launching, monitoring) ever
mutates it as a side effect -- a launch is a launch, a save is a
save.

Semantics:

* ``read``    -- the config as nested JSON (``TrainingConfig`` dump).
* ``read_raw`` -- the file's exact text (for a raw editor).
* ``update``  -- deep-merge ``overrides`` (a partial nested dict
  matching the config's structure) into the existing config,
  validate, write, return the result. The file must exist; unknown
  keys are ignored (``TrainingConfig`` is permissive by design).
* ``replace`` -- validate ``content`` as a full config document and
  write it, creating the file (and its directory) if needed.

Raises ``ConfigNotFoundError`` (file missing where one is required)
and ``ConfigInvalidError`` (parse/validation failure -- including a
failed validation of an update, which never touches the file).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ConfigFiles(ABC):
    @abstractmethod
    def read(self, config_path: Path) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def read_raw(self, config_path: Path) -> str:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, config_path: Path, overrides: dict[str, Any]
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def replace(self, config_path: Path, content: str) -> None:
        raise NotImplementedError
