"""ConfigInspector port -- read-side facts about a training config.

Two levels of detail:

* ``summarize`` -- the two facts every launch needs (mode, steps).
* ``describe``  -- everything the "continue from" picker needs: each
  start option's configured path and whether that target actually
  exists on disk right now.

Document CRUD lives in ``ConfigFiles``; the derived option schema in
``ConfigOptions``. This port never writes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ConfigSummary:
    """The two facts every start needs before the run row exists."""

    mode: str
    total_steps: int


@dataclass(frozen=True, slots=True)
class StartOption:
    """One entry in the "continue from" picker.

    ``path`` is what the config says (possibly relative, possibly
    empty); ``available`` is the truth after resolution -- the target
    exists on disk. A configured-but-missing target reports both, so
    a UI can show what it *would* use.
    """

    path: str
    available: bool
    label: str


@dataclass(frozen=True, slots=True)
class ConfigDescription:
    """Launch-relevant facts about one config file.

    ``start_from`` is keyed by launch value (``teacher`` / ``student``
    / ``resume``, plus ``lora_checkpoint`` when -- and only when -- the
    config is a LoRA config). Keys absent from the mapping are not
    options for this config at all; they are never fabricated as
    unavailable entries.
    """

    mode: str
    total_steps: int
    start_from: dict[str, StartOption]


class ConfigInspector(ABC):
    """Read + validate a config file.

    Raises ``ConfigNotFoundError`` / ``ConfigInvalidError``
    (application errors) -- the adapter translates parser failures, so
    use cases never touch the config format itself.
    """

    @abstractmethod
    def summarize(self, config_path: Path) -> ConfigSummary:
        raise NotImplementedError

    @abstractmethod
    def describe(self, config_path: Path) -> ConfigDescription:
        raise NotImplementedError
