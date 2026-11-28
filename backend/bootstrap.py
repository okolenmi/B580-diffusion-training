"""Composition root -- the one module allowed to import every layer.

Order of business:

1. ``Settings`` arrive fully built (from the CLI);
2. infrastructure objects are constructed (database + migrations,
   repository, event bus, clock, settings store, workspace layout,
   assets store, image-library and graph
   subsystems);
3. the supervisors and use cases are constructed with those ports;
4. ``ReconcileGraphExecutions`` sweeps rows
   left unfinished by a previous process (before any request can race
   them);
5. the aggregate goes to ``presentation.create_app``.

Steps 2-4 live in ``backend.wiring`` (one module per stage); this module
only orders them. Nothing else in the backend may know which concrete
classes exist -- that is what makes swapping an implementation (or a fake
in tests) a one-line change in the wiring and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .application.ports.clock import Clock
from .application.ports.journal import Journal
from .application.ports.monitor_bus import MonitorBus
from .application.graph_supervisor import GraphExecutionSupervisor
from .application.services import ApplicationServices
from .config import Settings
from .infrastructure.persistence.sqlite import SqliteDatabase
from .wiring.graph_kit import GraphKit
from .wiring.infrastructure import Infrastructure
from .wiring.services import build_services
from .wiring.startup import StartupSweep


@dataclass(frozen=True, slots=True)
class Container:
    """Wired infrastructure + application objects for one process."""

    settings: Settings
    database: SqliteDatabase
    clock: Clock
    graph_supervisor: GraphExecutionSupervisor
    services: ApplicationServices
    # The monitor adapter stays reachable for the few callers that
    # legitimately *publish* telemetry (tests; the graph runtime's
    # ExecutionContext wiring). Subscribing goes through
    # ``services.monitor.subscribe`` -- see docs 08 S-04.
    monitor_bus: MonitorBus
    #: What the consoles show; the entry point attaches the server's own log to it.
    journal: Journal


def build_container(settings: Settings) -> Container:
    """Wire the real implementation graph (idempotent for the DB)."""
    infra = Infrastructure.build(settings)
    kit = GraphKit.build(infra)
    services = build_services(infra, kit)
    StartupSweep(services, kit.sweep_scratch).run()
    return Container(
        settings=settings,
        database=infra.database,
        clock=infra.clock,
        graph_supervisor=kit.supervisor,
        services=services,
        monitor_bus=infra.monitor_bus,
        journal=infra.journal,
    )
