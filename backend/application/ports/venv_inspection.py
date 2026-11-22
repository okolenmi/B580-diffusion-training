"""Looking inside a virtualenv folder without importing anything from it.

The setup page asks which environment the trainer should use. For an existing one the answer is read here: the interpreter inside the
folder lists its installed distributions (metadata only), and the caller decides what is missing. Never writes.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class VenvDevice:
    present: bool
    name: str | None = None
    total_memory_mb: int | None = None
    torch_version: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class VenvInventory:
    interpreter: str | None
    python_version: str | None = None
    packages: tuple[tuple[str, str], ...] = ()
    error: str | None = None


def find_interpreter(venv_dir: str) -> str | None:
    """The interpreter of a virtualenv folder, given the folder or the interpreter file itself."""
    try:
        path = Path(venv_dir).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if path.is_file():
        return str(path) if path.name.lower().startswith("python") else None
    relative = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    candidate = path.joinpath(*relative)
    return str(candidate) if candidate.exists() else None


class VenvInspector(ABC):
    @abstractmethod
    def inspect(self, venv_dir: str) -> VenvInventory:
        """What the environment in ``venv_dir`` holds; an unreadable one is an ``error``, never an exception."""


    @abstractmethod
    def device(self, venv_dir: str) -> VenvDevice:
        """Whether torch in that environment can see the Intel GPU. Imports torch there, so it takes seconds."""


