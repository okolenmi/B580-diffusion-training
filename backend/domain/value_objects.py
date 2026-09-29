"""Value objects shared across the domain."""

from __future__ import annotations

from enum import Enum

RunId = int
"""Primary key of a persisted run.

Deliberately a plain ``int`` alias rather than a wrapper class: ids
cross SQL rows, path segments, and JSON bodies constantly, and a
runtime wrapper would buy ceremony without buying safety. Mypy-level
documentation still names the intent.
"""


class RunStatus(str, Enum):
    """Lifecycle states of a training run.

    ``created``   -- registered, process not launched yet
    ``running``   -- training process is alive
    ``completed`` -- process exited 0
    ``failed``    -- process exited non-zero or crashed
    ``cancelled`` -- stopped on request (the old server's
                     ``stopped``/``killed`` collapse into this one)
    """

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Terminal states never transition again."""
        return self in _TERMINAL


_TERMINAL = frozenset({RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})
