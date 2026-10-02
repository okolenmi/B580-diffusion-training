"""Application settings -- a frozen value object, loaded explicitly.

Nothing here touches the environment at import time: the CLI calls
``Settings.load()`` and hands the result to the composition root.
Every other module receives its configuration through constructor
injection; nothing in the backend reads ``os.environ`` directly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Pure path arithmetic (no I/O, no mutation) -- allowed at import.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Module-level defaults: a slots dataclass stores defaults only as slot
# descriptors, so `cls.port` would be a descriptor object, not 8766.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8766
_DEFAULT_DB_PATH = _PROJECT_ROOT / "backend" / "data" / "backend.db"


#: How a graph execution runs. ``child`` isolates it from the API server
#: process (WP-22: a device fault or an OOM kill takes the run down, not
#: the server, and a run survives a restart); ``inprocess`` runs the same
#: producer on a thread here. The values are the setting's wire shape, so
#: the composition root maps them straight onto a gateway rather than
#: re-parsing a string.
#:
#: ``child`` is the default because isolation is the whole point of
#: putting the run in another process, and the cost is measured: ~1.95s
#: of startup per run, against runs that take minutes. ``inprocess``
#: remains because it is the rollback -- if the child path breaks
#: something on hardware nobody tested, one variable undoes it.
GRAPH_EXECUTION_CHILD = "child"
GRAPH_EXECUTION_INPROCESS = "inprocess"
GRAPH_EXECUTION_MODES = (GRAPH_EXECUTION_CHILD, GRAPH_EXECUTION_INPROCESS)
#: Named once, because the field default, the env default and the
#: unrecognised-value fallback are three separate code paths and spelling
#: the mode in each of them is how a "default" ends up meaning two things.
DEFAULT_GRAPH_EXECUTION_MODE = GRAPH_EXECUTION_CHILD


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime configuration for one backend process."""

    project_root: Path
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    db_path: Path = _DEFAULT_DB_PATH
    #: Which gateway the supervisor gets. An env var rather than a row in
    #: the config table because it decides how this server process is
    #: built, not what the user configured about their project -- and
    #: because it must be readable before anything is wired, so a failed
    #: rollout is one variable away from being undone.
    graph_execution_mode: str = DEFAULT_GRAPH_EXECUTION_MODE

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from environment variables (defaults win).

        Recognised variables: ``BACKEND_HOST``, ``BACKEND_PORT``,
        ``BACKEND_DB_PATH``, ``BACKEND_GRAPH_EXECUTION``. The CLI layer
        may override individual fields on top of this via
        ``dataclasses.replace``.

        An unrecognised ``BACKEND_GRAPH_EXECUTION`` falls back to
        ``DEFAULT_GRAPH_EXECUTION_MODE`` instead of raising: this is read
        during start-up, and a typo in a convenience variable should not
        stop a server that is otherwise fine from starting.
        """
        env = os.environ if env is None else env
        db_path = env.get("BACKEND_DB_PATH")
        mode = env.get("BACKEND_GRAPH_EXECUTION", DEFAULT_GRAPH_EXECUTION_MODE)
        if mode not in GRAPH_EXECUTION_MODES:
            logger.warning(
                "ignoring BACKEND_GRAPH_EXECUTION=%r; expected one of %s",
                mode, ", ".join(GRAPH_EXECUTION_MODES),
            )
            mode = DEFAULT_GRAPH_EXECUTION_MODE
        return cls(
            project_root=_PROJECT_ROOT,
            host=env.get("BACKEND_HOST", _DEFAULT_HOST),
            port=int(env.get("BACKEND_PORT", str(_DEFAULT_PORT))),
            db_path=Path(db_path) if db_path else _DEFAULT_DB_PATH,
            graph_execution_mode=mode,
        )
