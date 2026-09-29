"""Application settings -- a frozen value object, loaded explicitly.

Nothing here touches the environment at import time: the CLI calls
``Settings.load()`` and hands the result to the composition root.
Every other module receives its configuration through constructor
injection; nothing in the backend reads ``os.environ`` directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# Pure path arithmetic (no I/O, no mutation) -- allowed at import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module-level defaults: a slots dataclass stores defaults only as slot
# descriptors, so `cls.port` would be a descriptor object, not 8766.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8766
_DEFAULT_DB_PATH = _PROJECT_ROOT / "backend" / "data" / "backend.db"


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration for one backend process."""

    project_root: Path
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    db_path: Path = _DEFAULT_DB_PATH

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables (defaults win).

        Recognised variables: ``BACKEND_HOST``, ``BACKEND_PORT``,
        ``BACKEND_DB_PATH``. The CLI layer may override individual
        fields on top of this via ``dataclasses.replace``.
        """
        env = os.environ if env is None else env
        db_path = env.get("BACKEND_DB_PATH")
        return cls(
            project_root=_PROJECT_ROOT,
            host=env.get("BACKEND_HOST", _DEFAULT_HOST),
            port=int(env.get("BACKEND_PORT", str(_DEFAULT_PORT))),
            db_path=Path(db_path) if db_path else _DEFAULT_DB_PATH,
        )
