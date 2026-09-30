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

# --------------------------------------------------------------------------
# Graph executions (M4)
# --------------------------------------------------------------------------

ExecutionId = int
"""Primary key of a persisted graph execution (same plain-int posture as
``RunId``: ids cross SQL rows, path segments, and JSON constantly)."""


class GraphStatus(str, Enum):
    """Lifecycle states of one node-graph execution.

    ``queued``   -- row created, worker thread not yet claimed it
    ``running``  -- worker claimed the row, nodes are building
    ``finished`` -- every node built successfully
    ``error``    -- graph-level failure or a node's build() raised
    ``stopped``  -- cancel requested (or a queued row was stopped
                    before its thread ever started)
    """

    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    STOPPED = "stopped"

    @property
    def is_terminal(self) -> bool:
        """Terminal states never transition again."""
        return self in _TERMINAL_GRAPH


_TERMINAL_GRAPH = frozenset(
    {GraphStatus.FINISHED, GraphStatus.ERROR, GraphStatus.STOPPED}
)
