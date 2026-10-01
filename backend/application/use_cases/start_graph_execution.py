"""StartGraphExecution -- validate, persist, launch one graph run.

Check-then-act under a lock (two concurrent starts must not both pass
the active-execution check); the row is added *before* the thread
starts so a crash between the two leaves something for startup
reconciliation to sweep.

Order of refusal mirrors ``StartDatasetTask``: the caller's own mistakes
first (validation errors -> 422 ``graph_invalid`` with the full issue
list), then the conflict (another execution lives -> 409
``graph_execution_active``). Single-active is a deliberate divergence
from the legacy endpoint's parallel runs -- one B580, and graph nodes
can build real training loops in-process (see doc 05 section 5).
"""

from __future__ import annotations

import threading

from ..dto import GraphExecutionSummaryDTO, to_execution_summary_dto
from ..errors import GraphExecutionActiveError, GraphInvalidError
from ..ports.clock import Clock
from ..ports.execution_launcher import ExecutionLauncher
from ..lifecycle_writer import ExecutionLifecycleWriter
from ..ports.graph_execution_repository import GraphExecutionRepository
from ..ports.graph_runtime import ISSUE_ERROR, GraphRuntime, issue_to_dict
from ...domain.entities.graph_execution import GraphExecution
from ...domain.events import DomainEvent
from ...domain.graph import GraphDefinition


class StartGraphExecution:
    """The only place a graph execution is born (single-active)."""

    def __init__(
        self,
        *,
        executions: GraphExecutionRepository,
        writer: ExecutionLifecycleWriter,
        runtime: GraphRuntime,
        launcher: ExecutionLauncher,
        clock: Clock,
    ) -> None:
        self._executions = executions
        self._writer = writer
        self._runtime = runtime
        self._launcher = launcher
        self._clock = clock
        self._lock = threading.Lock()

    def execute(self, graph: GraphDefinition) -> GraphExecutionSummaryDTO:
        with self._lock:
            issues = self._runtime.validate(graph)
            errors = [issue for issue in issues if issue.severity == ISSUE_ERROR]
            if errors:
                raise GraphInvalidError(
                    f"graph has {len(errors)} validation error(s)",
                    details=[issue_to_dict(issue) for issue in issues],
                )

            active = self._executions.find_active()
            if active is not None:
                raise GraphExecutionActiveError(
                    f"execution {active.id} is already {active.status.value}",
                    details={
                        "execution_id": active.id,
                        "status": active.status.value,
                    },
                )

            execution = GraphExecution.create(
                graph=graph, created_at=self._clock.now()
            )
            self._writer.insert(execution)  # binds id, buffers Queued, announces
            self._launcher.launch(execution.id, graph)
            return to_execution_summary_dto(execution)

