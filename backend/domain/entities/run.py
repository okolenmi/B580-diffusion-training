"""Run -- one training run, modelled as a state machine.

The entity owns its invariants:

* status only moves along ``RUN_TRANSITIONS`` -- anything else raises
  ``InvalidTransitionError``;
* progress may only be recorded while the run is ``running``;
* terminal states are final (derived from the table, not restated);
* every accepted mutation appends a domain event to an internal buffer,
  drained by the application layer via ``collect_events()`` after the
  entity is persisted;
* the plan never shrinks: ``total_steps`` only ever grows.

Every field is private behind a read-only property. That is the point:
``run.done_steps = -5`` or ``run.finished_at = None`` on a completed run
was legal Python that no check could stop, while the class docstring
claimed the entity owned its invariants (docs 08 S-11). The mutator
methods below are the only writers; persistence and the DTO layer read
through the properties and are unaffected.

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
from ..exceptions import DomainError
from ..lifecycle import StatusMachine
from ..value_objects import RUN_TRANSITIONS, RunId, RunStatus


class Run:
    """A training run with enforced lifecycle rules.

    Two ways in, on purpose:

    * ``create`` for a new run -- the only path that emits ``RunCreated``;
    * ``restore`` for a row loaded from the database -- which validates
      the cross-field rules a single column cannot (a running run with
      no ``started_at``, ``done_steps`` past ``total_steps``). The
      public ``__init__`` stays for the two of them, but callers outside
      this module should not be reaching for it (docs 08 S-16).
    """

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

        self._life: StatusMachine[RunStatus] = StatusMachine(
            transitions=RUN_TRANSITIONS, status=status, label="run", entity_id=id
        )
        self._config_path = config_path
        self._mode = mode
        self._total_steps = total_steps
        self._done_steps = done_steps
        self._current_loss = current_loss
        self._avg_loss = avg_loss
        self._phase = phase
        self._cache_done = cache_done
        self._cache_total = cache_total
        self._pid = pid
        self._exit_code = exit_code
        self._error = error
        self._log_path = log_path
        self._created_at = created_at
        self._updated_at = updated_at if updated_at is not None else created_at
        self._started_at = started_at
        self._finished_at = finished_at

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

    @classmethod
    def restore(cls, **fields: object) -> Run:
        """Rebuild a run from a persisted row, cross-field rules checked.

        The constructor validates single fields; a *loaded* aggregate can
        still be impossible as a whole (a running run with no
        ``started_at``, more steps done than planned, a terminal run
        with no ``finished_at``). Those rules are checked here, once,
        where they can be reported against the row that broke them.
        """
        status = fields.get("status")
        run = cls(**fields)  # type: ignore[arg-type] -- the mapper's kwargs
        assert isinstance(status, RunStatus) or isinstance(status, str)
        run._require_consistent(RunStatus(status))
        return run

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def id(self) -> RunId | None:
        """Persisted id, or ``None`` until the repository assigns one."""
        return self._life.entity_id  # type: ignore[return-value]

    @property
    def status(self) -> RunStatus:
        """Current lifecycle state."""
        return self._life.status

    @property
    def config_path(self) -> str:
        return self._config_path

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def done_steps(self) -> int:
        return self._done_steps

    @property
    def current_loss(self) -> float | None:
        return self._current_loss

    @property
    def avg_loss(self) -> float | None:
        return self._avg_loss

    @property
    def phase(self) -> str | None:
        return self._phase

    @property
    def cache_done(self) -> int | None:
        return self._cache_done

    @property
    def cache_total(self) -> int | None:
        return self._cache_total

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def exit_code(self) -> int | None:
        return self._exit_code

    @property
    def error(self) -> str | None:
        """Failure text, or the termination note for a cancelled run."""
        return self._error

    @property
    def log_path(self) -> str | None:
        return self._log_path

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at

    @property
    def started_at(self) -> datetime | None:
        return self._started_at

    @property
    def finished_at(self) -> datetime | None:
        return self._finished_at

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def assign_id(self, run_id: RunId) -> None:
        """Bind a persisted id; emits ``RunCreated``.

        Called by the repository right after the INSERT. Exactly once:
        re-binding raises, because the run's events would otherwise
        claim a different identity than they were emitted with.
        """
        self._life.bind(
            run_id,
            RunCreated(
                run_id=run_id,
                config_path=self._config_path,
                mode=self._mode,
                total_steps=self._total_steps,
                occurred_at=self._created_at,
            ),
        )

    def mark_started(self, *, pid: int | None, at: datetime) -> None:
        """Process launched: ``created`` -> ``running``."""
        run_id = self._require_id()  # before any mutation: fail cleanly
        self._life.move(RunStatus.RUNNING)
        self._pid = pid
        self._started_at = at
        self._stamp(at)
        self._life.buffer(RunStarted(run_id=run_id, pid=pid, occurred_at=at))

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

        ``None`` means "leave unchanged" for every optional field.
        ``total_steps`` is adopted explicitly by the caller (the
        supervisor only ever passes a *larger* total than the config
        promised) -- and it may only grow, which is why the rule is
        here rather than in the supervisor: a second writer calling this
        would otherwise be able to shrink the plan (docs 08 S-15).

        High-frequency telemetry deliberately does not emit domain
        events -- the supervisor publishes ``RunProgressed`` itself;
        domain events mark lifecycle changes only.
        """
        # Everything is checked before anything is written. A sample that
        # carried both a valid step count and an impossible total used to
        # leave the step count applied and the call rejected, which is
        # how a "refused" progress update still moved the run forward.
        self._life.require(RunStatus.RUNNING, action="record progress")
        if done_steps < 0:
            raise DomainError("done_steps cannot be negative")
        if total_steps is not None and total_steps < 0:
            raise DomainError("total_steps cannot be negative")
        if total_steps is not None and total_steps < self._total_steps:
            raise DomainError(
                f"total_steps cannot shrink ({self._total_steps} -> {total_steps})"
            )
        for value, label in ((cache_done, "cache_done"), (cache_total, "cache_total")):
            if value is not None and value < 0:
                raise DomainError(f"{label} cannot be negative")
        self._require_id()

        self._done_steps = done_steps
        if current_loss is not None:
            self._current_loss = current_loss
        if avg_loss is not None:
            self._avg_loss = avg_loss
        if phase is not None:
            self._phase = phase
        if total_steps is not None:
            self._total_steps = total_steps
        if cache_done is not None:
            self._cache_done = cache_done
        if cache_total is not None:
            self._cache_total = cache_total
        self._updated_at = at

    def mark_completed(self, *, at: datetime) -> None:
        """Process exited cleanly: ``running`` -> ``completed``."""
        run_id = self._require_id()
        self._life.move(RunStatus.COMPLETED)
        self._stamp(at)
        self._life.buffer(
            RunCompleted(run_id=run_id, done_steps=self._done_steps, occurred_at=at)
        )

    def mark_failed(
        self, *, at: datetime, error: str | None = None, exit_code: int | None = None
    ) -> None:
        """Process died: ``running`` -> ``failed``."""
        run_id = self._require_id()
        self._life.move(RunStatus.FAILED)
        self._error = error
        self._exit_code = exit_code
        self._stamp(at)
        self._life.buffer(
            RunFailed(run_id=run_id, error=error, exit_code=exit_code, occurred_at=at)
        )

    def cancel(self, *, at: datetime, reason: str | None = None) -> None:
        """Stopped on request: ``created``/``running`` -> ``cancelled``.

        ``reason`` (e.g. "stop requested (force)", "orphan cleanup: ...")
        is recorded in ``error`` -- for a cancelled run that field is the
        termination note, not a failure.
        """
        run_id = self._require_id()
        self._life.move(RunStatus.CANCELLED)
        if reason is not None:
            self._error = reason
        self._stamp(at)
        self._life.buffer(RunCancelled(run_id=run_id, reason=reason, occurred_at=at))

    # ------------------------------------------------------------------
    # Event buffer
    # ------------------------------------------------------------------

    def collect_events(self) -> list[DomainEvent]:
        """Drain and return buffered events (second call returns [])."""
        return self._life.drain()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _stamp(self, at: datetime) -> None:
        """Timestamps belong to the aggregate, not the machine: a move
        always touches ``updated_at``, and reaching a state with no way
        out also closes ``finished_at``."""
        self._updated_at = at
        if self._life.ends_lifecycle(self.status):
            self._finished_at = at

    def _require_consistent(self, status: RunStatus) -> None:
        """Cross-field rules for a *loaded* row (see ``restore``)."""
        if status is RunStatus.RUNNING and self._started_at is None:
            raise DomainError(f"run {self.id}: running but started_at is null")
        if status.is_terminal and self._finished_at is None:
            raise DomainError(
                f"run {self.id}: {status.value} but finished_at is null"
            )
        if status is not RunStatus.RUNNING and self._done_steps > self._total_steps:
            raise DomainError(
                f"run {self.id}: done_steps {self._done_steps} exceeds "
                f"total_steps {self._total_steps}"
            )

    def _require_id(self) -> RunId:
        return self._life.require_id()  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (
            f"Run(id={self._life.where}, status={self.status.value}, "
            f"{self._done_steps}/{self._total_steps} steps, mode={self._mode!r})"
        )