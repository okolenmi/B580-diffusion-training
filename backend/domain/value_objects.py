"""Value objects shared across the domain.

Each lifecycle enum carries its own transition table, and terminal-ness
is *derived* from that table rather than restated beside it. Two
separate lists would disagree by construction: a state terminal in one
could still be allowed to move in the other.

    RUN_TRANSITIONS[RunStatus.CREATED]  -- where a created run may go
    not RUN_TRANSITIONS[some_status]    -- is it terminal?

The machine that enforces the table is ``domain.lifecycle.StatusMachine``;
the table itself stays next to the enum it describes, because that is
where a reader looks for "what are the states".
"""

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
        """Terminal states never transition again (derived, see below)."""
        return not RUN_TRANSITIONS[self]



RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    # FAILED is reachable from CREATED: a launch can fail before the
    # process ever starts (missing interpreter, unreadable config), and
    # startup reconciliation fails runs abandoned mid-launch.
    RunStatus.CREATED: frozenset(
        {RunStatus.RUNNING, RunStatus.CANCELLED, RunStatus.FAILED}
    ),
    RunStatus.RUNNING: frozenset(
        {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

# --------------------------------------------------------------------------
# Graph executions (M4)
# --------------------------------------------------------------------------

ExecutionId = int
"""Primary key of a persisted graph execution (same plain-int posture as
``RunId``: ids cross SQL rows, path segments, and JSON constantly)."""


class GraphStatus(str, Enum):
    """Lifecycle states of one node-graph execution.

    ``planned``  -- accepted but waiting for room: no claim, no child; the
                    scheduler admits it when memory allows
    ``queued``   -- row created, worker thread not yet claimed it
    ``running``  -- worker claimed the row, nodes are building
    ``finished`` -- every node built successfully
    ``error``    -- graph-level failure or a node's build() raised
    ``stopped``  -- cancel requested (or a queued row was stopped
                    before its thread ever started)
    """

    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    FINISHED = "finished"
    ERROR = "error"
    STOPPED = "stopped"

    @property
    def is_terminal(self) -> bool:
        """Terminal states never transition again (derived, see below)."""
        return not GRAPH_TRANSITIONS[self]


GRAPH_TRANSITIONS: dict[GraphStatus, frozenset[GraphStatus]] = {
    GraphStatus.PLANNED: frozenset(
        {GraphStatus.QUEUED, GraphStatus.STOPPED, GraphStatus.ERROR}
    ),
    # ERROR is reachable from QUEUED: startup reconciliation fails rows a
    # dead process never got to, and validation-layer surprises can fail a
    # row before its thread claims it.
    GraphStatus.QUEUED: frozenset(
        {GraphStatus.RUNNING, GraphStatus.STOPPED, GraphStatus.ERROR}
    ),
    GraphStatus.RUNNING: frozenset(
        {GraphStatus.FINISHED, GraphStatus.ERROR, GraphStatus.STOPPED}
    ),
    GraphStatus.FINISHED: frozenset(),
    GraphStatus.ERROR: frozenset(),
    GraphStatus.STOPPED: frozenset(),
}