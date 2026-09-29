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
from .use_cases.get_run import GetRun
from .use_cases.list_runs import ListRuns


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    list_runs: ListRuns
    get_run: GetRun
    delete_runs: DeleteRuns
    event_bus: EventBus
