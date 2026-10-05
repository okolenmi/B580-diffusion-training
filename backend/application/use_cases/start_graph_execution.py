"""StartGraphExecution -- validate, admit, persist, launch one graph run.

Check-then-act under a lock (two concurrent starts must not both pass
the active-execution check); the admission claim and the row are both
taken inside that lock, *before* the thread starts, so a crash between
any of the steps leaves something for startup reconciliation to sweep.

Order of refusal mirrors ``StartDatasetTask``: the caller's own mistakes
first (validation errors -> 422 ``graph_invalid`` with the full issue
list), then the conflict (another execution lives -> 409
``graph_execution_active``), then admission (the device cannot fit it
-> 409 ``memory_unavailable`` with the breakdown; a refusal writes no
row and holds nothing). Single-active is a deliberate divergence
from the legacy endpoint's parallel runs -- one B580, and graph nodes
can build real training loops in-process (see doc 05 section 5).
"""

from __future__ import annotations

import threading

from ..dto import GraphExecutionSummaryDTO, to_execution_summary_dto
from ..errors import GraphExecutionActiveError, GraphInvalidError
from ..graph_peak_source import PeakSource
from ..memory_admission import LedgerSource, admit, graph_owner, pending_owner
from ..ports.clock import Clock
from ..ports.execution_launcher import ExecutionLauncher
from ..lifecycle_writer import ExecutionLifecycleWriter
from ..ports.graph_execution_repository import GraphExecutionRepository
from ..ports.graph_runtime import GraphRuntime, IssueSeverity, issue_to_dict
from ...domain.entities.graph_execution import GraphExecution
from ...domain.graph import GraphDefinition
from ...domain.memory_settings import effective_memory


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
        memory_ledger: LedgerSource,
        peak_source: PeakSource,
    ) -> None:
        self._executions = executions
        self._writer = writer
        self._runtime = runtime
        self._launcher = launcher
        self._clock = clock
        # Required, never defaulted: a container that starts executions
        # must say where admission lives (a lambda is enough). The
        # provider may still answer None -- the device total unknown --
        # and then every start is refused explicitly rather than
        # admitted unchecked.
        self._memory_ledger = memory_ledger
        # Required for the same reason: where this graph's remembered
        # peak comes from. The answer may be "unknown" for every graph
        # (``unknown_peaks``), but the container says so deliberately
        # rather than by omission.
        self._peak_source = peak_source
        self._lock = threading.Lock()

    def execute(
        self,
        graph: GraphDefinition,
        *,
        memory_overrides: dict | None = None,
    ) -> GraphExecutionSummaryDTO:
        """Admit and launch one run.

        ``memory_overrides`` is the execution request's own copy of the
        graph's memory settings (MEM-02 #2): the effective values are
        computed here, once, against the ledger's capacity, and stored
        on the row; the ledger's claim (``reserved_mb``) is stored with
        them, so a restart reproduces the same held total from the
        rows. The observed half of the demand (MEM-04 #2) comes from
        ``peak_source``: a known fingerprint with a remembered peak makes
        the demand ``peak + pillow`` (stated still beats observed), and
        the fingerprint key travels onto the row so the watcher can file
        this run's own reported peak under the same key.
        """
        ledger = self._memory_ledger()
        with self._lock:
            issues = self._runtime.validate(graph)
            errors = [issue for issue in issues if IssueSeverity(issue.severity).blocks]
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

            observed = self._peak_source(graph)
            memory = effective_memory(
                graph.memory,
                memory_overrides,
                peak_record=observed.peak_record(),
                fingerprint_key=observed.fingerprint_key,
                capacity_mb=ledger.capacity_mb if ledger is not None else None,
            )
            exploratory = memory.demand_source == "unknown"
            if ledger is None:
                # No ledger: admit() refuses with device-total-unknown
                # before it looks at the demand (rule 2).
                device_demand = 0.0
            elif exploratory or memory.demand_mb is None:
                # Unknown demand claims all free capacity as an
                # exploratory exclusive run; a None demand cannot
                # happen while a ledger exists (its capacity is what
                # `unknown` falls back to) and never becomes a zero
                # claim here either.
                device_demand = (
                    memory.demand_mb
                    if memory.demand_mb is not None
                    else ledger.capacity_mb
                )
            else:
                # Stated/observed are allocator MB; a grant is device MB
                # (task rule 4: + the per-process overhead).
                device_demand = memory.demand_mb + ledger.process_overhead_mb

            provisional = pending_owner("graph")
            grant = admit(
                ledger,
                provisional,
                device_demand,
                exploratory=exploratory,
                what="this graph execution",
            )
            execution = GraphExecution.create(
                graph=graph,
                created_at=self._clock.now(),
                memory=memory,
                reserved_mb=grant.mb,
            )
            try:
                self._writer.insert(execution)  # binds id, buffers Queued, announces
                if ledger is not None:
                    ledger.rename(provisional, graph_owner(execution.require_id()))
                self._launcher.launch(execution.require_id(), graph)
            except BaseException:
                # Failed between the claim and the child: nothing stays
                # held. Both owner forms go, because release is
                # idempotent and the rename may or may not have happened.
                # A hard crash cannot run this -- there, the queued row's
                # claim is rebuilt at startup and released by reconcile
                # when the row turns out to have no child.
                if ledger is not None:
                    if execution.id is not None:
                        ledger.release(graph_owner(execution.id))
                    ledger.release(provisional)
                raise
            return to_execution_summary_dto(execution)

