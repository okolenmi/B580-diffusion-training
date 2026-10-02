"""GetStartOptions -- data for the "continue from" picker.

Answers three questions for one config file:

1. what would each start option use (``start_from`` map, only keys
   that exist for this config, ``available`` = target exists now);
2. is a run currently active (``has_unfinished_run`` -- the same
   ``find_active`` invariant that blocks a second launch);
3. what did the most recent finished run look like (newest-first
   scan of a small window; ``None`` when nothing finished yet).

A broken config raises the config errors -- the caller must see why
a launch would fail rather than an empty picker.
"""

from __future__ import annotations


from ..dto import LastFinishedRun, StartOptionsResult
from ..ports.config_inspector import ConfigInspector
from ..project_paths import ProjectPaths
from ..ports.run_repository import RunRepository


class GetStartOptions:
    def __init__(
        self, *, inspector: ConfigInspector, runs: RunRepository, paths: ProjectPaths
    ) -> None:
        self._inspector = inspector
        self._runs = runs
        self._paths = paths

    def execute(self, config_path: str) -> StartOptionsResult:
        description = self._inspector.describe(self._paths.config(config_path))

        active = self._runs.find_active()
        last_finished: LastFinishedRun | None = None
        for run in self._runs.list_runs(limit=10):
            if run.status.is_terminal:
                last_finished = LastFinishedRun(
                    id=run.require_id(),
                    config_path=run.config_path,
                    mode=run.mode,
                    done_steps=run.done_steps,
                    total_steps=run.total_steps,
                    avg_loss=run.avg_loss,
                    status=run.status.value,
                )
                break

        return StartOptionsResult(
            start_from=description.start_from,
            has_unfinished_run=active is not None,
            last_finished=last_finished,
        )
