"""Run -- one training run, modelled as a state machine.

The entity owns its invariants:

* status only moves along the transition table below -- anything else
  raises ``InvalidTransitionError`` (``run.status`` is exposed read-only;
  there is no setter that bypasses the machine);
* progress may only be recorded while the run is ``running``;
* terminal states are final;
* every accepted mutation appends a domain event to an internal buffer,
  drained by the application layer via ``collect_events()`` after the
  entity is persisted.

Timestamps are always passed *in* (``at=...``) rather than read from a
clock, so tests are deterministic and the entity stays pure.
"""

from __future__ import annotations

from datetime import datetime

from ..events import (
    DomainEvent,
    RunCancelled,
    RunCompleted,
    RunCreated,
    RunFailed,
    RunStarted,
)
from ..exceptions import DomainError, InvalidTransitionError
from ..value_objects import RunId, RunStatus


class Run:
    """A training run with enforced lifecycle rules."""

    _ALLOWED: dict[RunStatus, frozenset[RunStatus]] = {
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

    def __init__(
        self,
        *,
        status: RunStatus,
        config_path: str,
        mode: str,
        created_at: datetime,
        updated_at: datetime | None = None,
        id: RunId | None = None,
        total_steps: int = 0,
        done_steps: int = 0,
        current_loss: float | None = None,
        avg_loss: float | None = None,
        phase: str | None = None,
        pid: int | None = None,
        exit_code: int | None = None,
        error: str | None = None,
        log_path: str | None = None,
        cache_done: int | None = None,
        cache_total: int | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
    ) -> None:
        if not config_path:
            raise DomainError("config_path is required")
        if not mode:
            raise DomainError("mode is required")
        if total_steps < 0:
            raise DomainError("total_steps cannot be negative")
        if done_steps < 0:
            raise DomainError("done_steps cannot be negative")
        if isinstance(status, str) and not isinstance(status, RunStatus):
            status = RunStatus(status)

        self._id: RunId | None = id
        self._status: RunStatus = status
        self._events: list[DomainEvent] = []

        self.config_path = config_path
        self.mode = mode
        self.total_steps = total_steps
        self.done_steps = done_steps
        self.current_loss = current_loss
        self.avg_loss = avg_loss
        self.phase = phase
        self.cache_done = cache_done
        self.cache_total = cache_total
        self.pid = pid
        self.exit_code = exit_code
        self.error = error
        self.log_path = log_path
        self.created_at = created_at
        self.updated_at = updated_at if updated_at is not None else created_at
        self.started_at = started_at
        self.finished_at = finished_at

    # ------------------------------------------------------------------
    # Construction / identity
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        config_path: str,
        mode: str,
        total_steps: int,
        created_at: datetime,
    ) -> Run:
        """Register a new run in ``created`` state (no id yet)."""
        return cls(
            status=RunStatus.CREATED,
            config_path=config_path,
            mode=mode,
            total_steps=total_steps,
            created_at=created_at,
        )

    @property
    def id(self) -> RunId | None:
        """Persisted id, or ``None`` until the repository assigns one."""
        return self._id

    @property
    def status(self) -> RunStatus:
        """Current lifecycle state (read-only -- no setter exists)."""
        return self._status

    def assign_id(self, run_id: RunId) -> None:
        """Bind a persisted id; emits ``RunCreated``.

        Called by the repository right after the INSERT. Exactly once:
        re-binding raises, because the run's events would otherwise
        claim a different identity than they were emitted with.
        """
        if self._id is not None:
            raise DomainError(f"run already has id {self._id}")
        if run_id < 1:
            raise DomainError("run id must be a positive integer")
        self._id = run_id
        self._events.append(
            RunCreated(
                run_id=run_id,
                config_path=self.config_path,
                mode=self.mode,
                total_steps=self.total_steps,
                occurred_at=self.created_at,
            )
        )

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def mark_started(self, *, pid: int | None, at: datetime) -> None:
        """Process launched: ``created`` -> ``running``."""
        run_id = self._require_id()  # before any mutation: fail cleanly
        self._transition(to=RunStatus.RUNNING, at=at)
        self.pid = pid
        self.started_at = at
        self._emit(RunStarted(run_id=run_id, pid=pid, occurred_at=at))

    def record_progress(
        self,
        *,
        done_steps: int,
        at: datetime,
        current_loss: float | None = None,
        avg_loss: float | None = None,
        phase: str | None = None,
        total_steps: int | None = None,
        cache_done: int | None = None,
        cache_total: int | None = None,
    ) -> None:
        """Update training progress (``running`` only, no event).

        ``None`` means "leave unchanged" for every optional field;
        ``total_steps`` is adopted explicitly by the caller (the
        supervisor only ever passes a *larger* total than the config
        promised). High-frequency telemetry deliberately does not emit
        domain events -- the supervisor publishes ``RunProgressed``
        itself; domain events mark lifecycle changes only.
        """
        self._require_status(RunStatus.RUNNING, action="record progress")
        if done_steps < 0:
            raise DomainError("done_steps cannot be negative")
        if total_steps is not None and total_steps < 0:
            raise DomainError("total_steps cannot be negative")
        for value, label in ((cache_done, "cache_done"), (cache_total, "cache_total")):
            if value is not None and value < 0:
                raise DomainError(f"{label} cannot be negative")
        self._require_id()
        self.done_steps = done_steps
        if current_loss is not None:
            self.current_loss = current_loss
        if avg_loss is not None:
            self.avg_loss = avg_loss
        if phase is not None:
            self.phase = phase
        if total_steps is not None:
            self.total_steps = total_steps
        if cache_done is not None:
            self.cache_done = cache_done
        if cache_total is not None:
            self.cache_total = cache_total
        self.updated_at = at

    def mark_completed(self, *, at: datetime) -> None:
        """Process exited cleanly: ``running`` -> ``completed``."""
        run_id = self._require_id()
        self._transition(to=RunStatus.COMPLETED, at=at)
        self._emit(RunCompleted(run_id=run_id, done_steps=self.done_steps, occurred_at=at))

    def mark_failed(
        self, *, at: datetime, error: str | None = None, exit_code: int | None = None
    ) -> None:
        """Process died: ``running`` -> ``failed``."""
        run_id = self._require_id()
        self._transition(to=RunStatus.FAILED, at=at)
        self.error = error
        self.exit_code = exit_code
        self._emit(
            RunFailed(run_id=run_id, error=error, exit_code=exit_code, occurred_at=at)
        )

    def cancel(self, *, at: datetime, reason: str | None = None) -> None:
        """Stopped on request: ``created``/``running`` -> ``cancelled``.

        ``reason`` (e.g. "stop requested (force)", "orphan cleanup: ...")
        is recorded in ``error`` -- for a cancelled run that field is the
        termination note, not a failure.
        """
        run_id = self._require_id()
        self._transition(to=RunStatus.CANCELLED, at=at)
        if reason is not None:
            self.error = reason
        self._emit(RunCancelled(run_id=run_id, reason=reason, occurred_at=at))

    # ------------------------------------------------------------------
    # Event buffer
    # ------------------------------------------------------------------

    def collect_events(self) -> list[DomainEvent]:
        """Drain and return buffered events (second call returns [])."""
        drained = self._events
        self._events = []
        return drained

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition(self, *, to: RunStatus, at: datetime) -> None:
        allowed = self._ALLOWED[self._status]
        if to not in allowed:
            where = self._id if self._id is not None else "<new>"
            raise InvalidTransitionError(
                f"run {where}: {self._status.value} -> {to.value} is not allowed"
            )
        self._status = to
        self.updated_at = at
        if to.is_terminal:
            self.finished_at = at

    def _require_status(self, status: RunStatus, *, action: str) -> None:
        if self._status is not status:
            raise InvalidTransitionError(
                f"run {self._id if self._id is not None else '<new>'}: "
                f"cannot {action} while {self._status.value} (needs {status.value})"
            )

    def _require_id(self) -> RunId:
        if self._id is None:
            raise DomainError("run has no id yet -- persist it first")
        return self._id

    def _emit(self, event: DomainEvent) -> None:
        self._events.append(event)

    def __repr__(self) -> str:
        where = self._id if self._id is not None else "<new>"
        return (
            f"Run(id={where}, status={self._status.value}, "
            f"{self.done_steps}/{self.total_steps} steps, mode={self.mode!r})"
        )
