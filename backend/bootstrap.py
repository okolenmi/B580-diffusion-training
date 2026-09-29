"""Composition root -- the one module allowed to import every layer.

Order of business:

1. ``Settings`` arrive fully built (from the CLI);
2. infrastructure objects are constructed (database + migrations,
   repository, event bus, clock);
3. use cases are constructed with those ports;
4. the aggregate goes to ``presentation.create_app``.

Nothing else in the backend may know which concrete classes exist --
that is what makes swapping an implementation (or a fake in tests) a
one-line change here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .application.ports.clock import Clock
from .application.services import ApplicationServices
from .application.use_cases import DeleteRuns, GetRun, ListRuns
from .config import Settings
from .infrastructure.clock import SystemClock
from .infrastructure.events.callback_event_bus import CallbackEventBus
from .infrastructure.persistence.run_repository import SqliteRunRepository
from .infrastructure.persistence.sqlite import SqliteDatabase


@dataclass(frozen=True, slots=True)
class Container:
    """Wired infrastructure + application objects for one process."""

    settings: Settings
    database: SqliteDatabase
    clock: Clock
    services: ApplicationServices


def build_container(settings: Settings) -> Container:
    """Wire the real implementation graph (idempotent for the DB)."""
    database = SqliteDatabase(settings.db_path)
    database.initialize()

    run_repository = SqliteRunRepository(database)
    event_bus = CallbackEventBus()
    clock = SystemClock()  # consumed by M2's StartTraining use case

    services = ApplicationServices(
        list_runs=ListRuns(run_repository),
        get_run=GetRun(run_repository),
        delete_runs=DeleteRuns(run_repository, event_bus),
        event_bus=event_bus,
    )
    return Container(
        settings=settings,
        database=database,
        clock=clock,
        services=services,
    )
