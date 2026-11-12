"""StartGraphExecution -- validate, admit, persist, launch graph runs (several may live at once).

Check-then-act under a lock (two concurrent starts must not both claim the same free memory); the admission claim
and the row are both taken inside that lock, *before* the thread starts, so a crash between any of the steps leaves
something for startup reconciliation to sweep.

Order of refusal mirrors ``StartDatasetTask``: the caller's own mistakes first (validation errors -> 422
``graph_invalid`` with the full issue list), then admission (the device cannot fit it -> 409 ``memory_unavailable``
with the breakdown; a refusal writes no row and holds nothing). Memory admission is the only gate on parallel runs.

What a start does when there is no room depends on its ``mode`` (the editor's Run button remembers one):

* ``parallel`` -- start now or refuse with the breakdown;
* ``auto`` -- start now if admitted, otherwise store a ``planned`` row (no claim, no child);
* ``queue`` -- start only when nothing is live or planned, otherwise ``planned``.

Planned rows start in order of creation, from ``promote_planned``, which the pump calls whenever memory may have
changed. The head of the line is never overtaken by a later row, so a big run is not starved by small ones.
"""

from __future__ import annotations

import threading

from ..dto import GraphExecutionSummaryDTO, to_execution_summary_dto
from ..errors import GraphInvalidError, MemoryUnavailableError
from ..execution_admission import ExecutionAdmission
from ..graph_peak_source import PeakSource
from ..memory_admission import LedgerSource, graph_owner
from .rename_graph_execution import clean_name
from ..ports.clock import Clock
from ..ports.execution_launcher import ExecutionLauncher
from ..lifecycle_writer import ExecutionLifecycleWriter
from ..ports.graph_execution_repository import GraphExecutionRepository
from ..ports.graph_runtime import GraphRuntime, IssueSeverity, issue_to_dict
from ...domain.entities.graph_execution import GraphExecution
from ...domain.graph import GraphDefinition
from ...domain.value_objects import GraphStatus
from ...domain.memory_settings import EffectiveMemory


MODES = ("auto", "parallel", "queue")


class StartGraphExecution:
    """The only place a graph execution is born, and where a planned one is promoted."""

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
        self._admission = ExecutionAdmission(ledger=memory_ledger, peak_source=peak_source)
        self._lock = threading.Lock()

    def execute(
        self,
        graph: GraphDefinition,
        *,
        memory_overrides: dict | None = None,
        mode: str = "auto",
        name: str | None = None,
    ) -> GraphExecutionSummaryDTO:
        """Admit and launch one run, or plan it (see the module docstring).

        ``memory_overrides`` is the execution request's own copy of the graph's memory settings (MEM-02 #2): the
        effective values are computed once against the ledger's capacity and stored on the row, with the ledger's claim
        (``reserved_mb``), so a restart reproduces the same held total from the rows. The observed half of the demand
        (MEM-04 #2) comes from ``peak_source``: a known fingerprint with a remembered peak makes the demand
        ``peak + pillow`` (stated still beats observed), and the fingerprint key travels onto the row so the watcher can
        file this run's own reported peak under the same key.
        """
        if mode not in MODES:
            raise ValueError(f"unknown start mode {mode!r}")
        with self._lock:
            issues = self._runtime.validate(graph)
            errors = [issue for issue in issues if IssueSeverity(issue.severity).blocks]
            if errors:
                raise GraphInvalidError(
                    f"graph has {len(errors)} validation error(s)",
                    details=[issue_to_dict(issue) for issue in issues],
                )
            memory = self._admission.memory_for(graph, memory_overrides)
            name = clean_name(name)
            if mode == "queue" and self._anything_live_or_planned():
                return self._plan(graph, memory, name)
            if mode != "parallel" and self._executions.list_planned():
                return self._plan(graph, memory, name)  # never overtake the line
            try:
                return self._launch(graph, memory, planned=None, name=name)
            except MemoryUnavailableError as refusal:
                if mode == "parallel" or not self._admission.worth_waiting(refusal):
                    raise
                return self._plan(graph, memory, name)

    def promote_planned(self) -> int:
        """Start planned rows, oldest first, while the next one is admitted. Returns how many started."""
        started = 0
        with self._lock:
            for execution in self._executions.list_planned():
                memory = self._admission.memory_for(execution.graph, self._settings_of(execution.memory))
                try:
                    self._launch(execution.graph, memory, planned=execution)
                except MemoryUnavailableError:
                    break
                started += 1
        return started

    def _anything_live_or_planned(self) -> bool:
        return self._executions.find_active() is not None or bool(self._executions.list_planned())

    @staticmethod
    def _settings_of(memory: EffectiveMemory | None) -> dict | None:
        """A planned row's budget settings as the overrides they were planned with (its demand is looked up again)."""
        if memory is None:
            return None
        return {
            "vram_min_mb": memory.vram_min_mb, "vram_max_mb": memory.vram_max_mb, "strict": memory.strict,
            "policy": memory.policy, "ram_max_mb": memory.ram_max_mb,
        }

    def _plan(self, graph: GraphDefinition, memory: EffectiveMemory, name: str | None) -> GraphExecutionSummaryDTO:
        execution = GraphExecution.plan(graph=graph, created_at=self._clock.now(), memory=memory, name=name)
        self._writer.insert(execution)
        return to_execution_summary_dto(execution)

    def _launch(
        self, graph: GraphDefinition, memory: EffectiveMemory, *, planned: GraphExecution | None, name: str | None = None
    ) -> GraphExecutionSummaryDTO:
        """Claim, persist (a new row, or the planned one moving to queued) and start the child."""
        claim = self._admission.claim(memory)
        execution = planned
        try:
            if planned is None:
                execution = GraphExecution.create(
                    graph=graph, created_at=self._clock.now(), memory=memory, reserved_mb=claim.grant.mb, name=name
                )
                self._writer.insert(execution)  # binds id, buffers Queued, announces
            else:
                planned.promote(at=self._clock.now(), memory=memory, reserved_mb=claim.grant.mb)
                if not self._writer.commit(planned, expected=GraphStatus.PLANNED):
                    self._admission.release(claim)  # cancelled between the read and the write
                    return to_execution_summary_dto(planned)
            self._admission.rename(claim, graph_owner(execution.require_id()))
            self._launcher.launch(execution.require_id(), graph)
        except BaseException:
            # Failed between the claim and the child: nothing stays held. A hard crash cannot run this -- there, the
            # queued row's claim is rebuilt at startup and released by reconcile when the row has no child.
            self._admission.release(claim, graph_owner(execution.id) if execution is not None and execution.id is not None else None)
            raise
        return to_execution_summary_dto(execution)
