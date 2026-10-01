"""StopGraphExecution -- cancel a queued/running execution.

Order mirrors ``StopTraining``: signal first (set the cancel event --
nodes poll it cooperatively between heavy steps, and the supervisor
checks it between nodes), then claim the terminal transition with a
CAS on the status read *before* mutating.

The bounded retry loop handles the one real race: the worker claiming
``queued -> running`` between our read and our write. If a terminal
writer wins outright (the run finished on its own a moment ago), the
honest response is 409 ``graph_execution_not_active`` naming that
winner -- never a silent mislabel.
"""

from __future__ import annotations

from ..dto import GraphExecutionDTO, to_execution_dto
from ..errors import (
    GraphExecutionNotFoundError,
    GraphExecutionNotActiveError,
)
from ..ports.clock import Clock
from ..ports.execution_launcher import ExecutionLauncher
from ..event_publisher import EventPublisher
from ..ports.graph_execution_repository import GraphExecutionRepository

_MAX_ATTEMPTS = 3


class StopGraphExecution:
    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        events: EventPublisher,
        launcher: ExecutionLauncher,
        clock: Clock,
    ) -> None:
        self._executions = executions
        self._events = events
        self._launcher = launcher
        self._clock = clock

    def execute(self, execution_id: int) -> GraphExecutionDTO:
        self._launcher.cancel(execution_id)  # signal, then claim

        for _ in range(_MAX_ATTEMPTS):
            execution = self._executions.get(execution_id)
            if execution is None:
                raise GraphExecutionNotFoundError(
                    f"execution {execution_id} does not exist"
                )
            if execution.status.is_terminal:
                raise GraphExecutionNotActiveError(
                    f"execution {execution_id} already {execution.status.value}",
                    details={"status": execution.status.value},
                )
            expected = execution.status
            execution.stop(at=self._clock.now(), reason="stop requested")
            if self._executions.update_if_status(execution, expected=expected):
                self._events.publish(execution)
                return to_execution_dto(execution)
            # CAS lost -- most likely the worker claimed queued->running
            # between our read and our write; refetch and try again.

        fresh = self._executions.get(execution_id)
        status = fresh.status.value if fresh else "?"
        raise GraphExecutionNotActiveError(
            f"execution {execution_id} kept racing to another state (now {status})",
            details={"status": status},
        )
