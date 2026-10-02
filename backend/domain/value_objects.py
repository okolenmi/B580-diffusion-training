"""Value objects shared across the domain.

Each lifecycle enum carries its own transition table, and terminal-ness
is *derived* from that table rather than restated beside it. The two
used to disagree by construction: adding a terminal state meant adding
it to ``_ALLOWED`` (empty row) and to a separate ``_TERMINAL`` set, and a
state that was terminal in one list was still allowed to move in the
other (docs 08 S-12).

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


class TrainingMode(str, Enum):
    """``tuning.method`` -- what a run actually trains.

    The four values are the ones ``nodes.config_model`` declares (four
    ``Literal`` unions, one per tuning strategy). ``Run.mode`` used to
    be a bare ``str``, so a typo or a fifth strategy reached the database
    as a run that could never be read back as anything meaningful; a
    ``str``-valued enum keeps every existing comparison and JSON
    round-trip working while making the vocabulary a type
    (docs 08 S-24).

    Not a ``training_mode`` naming: it *is* the tuning method, and the
    config inspector is what produces it.
    """

    LORA = "lora"
    CYCLIC = "cyclic"
    DISTILLATION = "distillation"
    FULL = "full"


class StartFrom(str, Enum):
    """What a launch starts from -- a checkpoint, a teacher run, or a
    previous run's own state.

    The values are exactly the keys ``ConfigInspector.describe`` returns
    in its ``start_from`` map (the options a config actually offers), so
    the launch vocabulary and the offered one cannot drift apart: the
    UI sends a key the inspector produced, or the use case refuses it
    with the list (docs 08 S-24).
    """

    TEACHER = "teacher"
    STUDENT = "student"
    RESUME = "resume"
    LORA_CHECKPOINT = "lora_checkpoint"


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
        """Terminal states never transition again (derived, see below)."""
        return not GRAPH_TRANSITIONS[self]


GRAPH_TRANSITIONS: dict[GraphStatus, frozenset[GraphStatus]] = {
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