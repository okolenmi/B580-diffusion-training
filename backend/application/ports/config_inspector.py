"""ConfigInspector port -- summarise a training config for a start."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConfigSummary:
    """The two facts every start needs before the run row exists."""

    mode: str
    total_steps: int


class ConfigInspector(ABC):
    """Read + validate a config file.

    Raises ``ConfigNotFoundError`` / ``ConfigInvalidError``
    (application errors) -- the adapter translates parser failures, so
    use cases never touch the config format itself.
    """

    @abstractmethod
    def summarize(self, config_path: Path) -> ConfigSummary:
        raise NotImplementedError
