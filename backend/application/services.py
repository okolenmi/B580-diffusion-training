"""ApplicationServices -- the wired-up use cases plus shared ports.

The composition root (``backend.bootstrap``) fills this frozen
aggregate and hands it to presentation; it is the *only* object the
web layer may reach for. Holding it in the application layer (rather
than in presentation) keeps the dependency direction honest: the web
layer depends on application, never the other way around.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ports.event_bus import EventBus
from .use_cases.delete_runs import DeleteRuns
from .use_cases.get_active_run import GetActiveRun
from .use_cases.get_run import GetRun
from .use_cases.get_run_log import GetRunLog
from .use_cases.list_runs import ListRuns
from .use_cases.reconcile_runs import ReconcileRuns
from .use_cases.start_training import StartTraining
from .use_cases.stop_training import StopTraining


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    list_runs: ListRuns
    get_run: GetRun
    delete_runs: DeleteRuns
    get_active_run: GetActiveRun
    start_training: StartTraining
    stop_training: StopTraining
    get_run_log: GetRunLog
    reconcile_runs: ReconcileRuns
    event_bus: EventBus
