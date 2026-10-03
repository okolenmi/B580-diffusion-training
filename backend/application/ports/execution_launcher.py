"""ExecutionLauncher port -- "run this graph in the background" as a dependency.

`GraphExecutionSupervisor` is the production implementation; the use cases
that start, stop and reconcile an execution reach it through these verbs.
Keeping them behind a port means the use cases no longer import a concrete
supervisor class (docs 08 S-01), and a test can assert on "was it launched
/ cancelled / is it still going" with a small recorder.

``launch``/``cancel`` are about starting and stopping. ``adopt`` and
``recorded_outcome`` are about the same run from the other side of a
restart: one asks whether it is still going, the other asks what it
already said about itself. Both are questions only the thing that owns the
run can answer, and the startup sweep needs both before deciding a row's
fate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ...domain.graph import GraphDefinition, NodeResult
from ...domain.value_objects import ExecutionId


@dataclass(frozen=True, slots=True)
class RecordedOutcome:
    """What a finished run already wrote down about itself.

    A run appends an outcome record as its last act, so this is the run's
    own account of how it ended -- not an inference from the fact that the
    process is gone. ``error is None`` means it succeeded.
    """

    results: tuple[NodeResult, ...]
    error: str | None


class ExecutionLauncher(ABC):
    """Start and cooperatively cancel graph execution threads."""

    @abstractmethod
    def launch(self, execution_id: ExecutionId, graph: GraphDefinition) -> None:
        """Run ``graph`` in the background for ``execution_id``."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, execution_id: ExecutionId) -> None:
        """Ask the running execution to stop; no-op when not running here.

        Whether the row actually stops is decided by the stop use case's
        status compare-and-swap, not by this call.
        """
        raise NotImplementedError

    @abstractmethod
    def adopt(self, execution_id: ExecutionId) -> int | None:
        """Re-attach to a run still going after a restart; None if none.

        Only meaningful for a supervisor whose runs live in their own
        process, and the honest answer from one whose runs are threads is
        ``None`` -- those die with the server, so there is nothing to find.
        """
        raise NotImplementedError

    @abstractmethod
    def recorded_outcome(self, execution_id: ExecutionId) -> RecordedOutcome | None:
        """What this run's own record says, or ``None`` if it never said.

        Distinct from ``adopt``, and the two do not exclude each other: a
        run can be still running (``adopt`` answers with a pid) *and* have
        written nothing yet (``recorded_outcome`` answers ``None``).

        ``None`` means the run's record is absent or has no outcome in it
        -- it was killed before finishing, or never got far enough to
        write. That is the case where absence is all the evidence there
        is, and the caller is entitled to read it as failure.

        This is why a run that finished while the server was down is not
        reported as a failure: its verdict and its results are already on
        disk, and reporting "crashed" over a clean outcome would discard
        work that succeeded.
        """
        raise NotImplementedError