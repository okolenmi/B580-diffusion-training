"""SubprocessGraphTaskGateway -- spawning graph-execution children.

The dataset-task gateway is the template and this is deliberately the same
shape: build a command, spawn it into its own session so one signal reaches
the whole group, and answer liveness through the *shared* marker check in
``infrastructure/process_identity.py`` rather than a fourth copy of it.

What is different is stop. The dataset gateway kills outright because a
task either finished ingesting or did not. A graph execution can be
stopped between two nodes and still leave a usable event file, a released
allocator and an accurate outcome record, so ``request_stop`` sends
SIGINT and lets the child's runtime notice at its next step boundary;
``kill`` remains for the escalation the supervisor performs when the
grace period expires.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path

from ..application.ports.graph_task_gateway import (
    GraphLaunchError,
    GraphTaskGateway,
    GraphTaskLaunch,
)
from .process_identity import cmdline_mentions, find_by_argv
from .workspace import WorkspaceLayout

logger = logging.getLogger(__name__)

#: What identifies one of our children, read back out of /proc. Chosen to
#: be a module path rather than a flag, because a flag value could
#: coincidentally appear in an unrelated process's argv while a module
#: path of ours essentially cannot.
CMDLINE_MARKER = "backend.infrastructure.graph_task_worker"


class SubprocessGraphTaskGateway(GraphTaskGateway):
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout
        self._procs: dict[int, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # -- spawn ------------------------------------------------------------

    def spawn(self, launch: GraphTaskLaunch) -> int:
        log_path = Path(launch.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "w", encoding="utf-8", buffering=1) as log:
                proc = subprocess.Popen(
                    self._build_command(launch),
                    cwd=str(self._layout.project_root),
                    env=self._build_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    # Its own session, so request_stop/kill reach the
                    # child's whole process group and cannot leak a
                    # signal to this server.
                    start_new_session=True,
                )
        except Exception as exc:
            raise GraphLaunchError(
                f"failed to launch graph execution child: {exc}"
            ) from exc
        with self._lock:
            self._procs[proc.pid] = proc
        return proc.pid

    def _build_command(self, launch: GraphTaskLaunch) -> list[str]:
        return [
            self._layout.venv_python,
            "-m",
            CMDLINE_MARKER,
            "--execution",
            str(launch.execution_id),
            "--graph",
            str(launch.graph_path),
            "--events",
            str(launch.event_path),
        ]

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        project_root = self._layout.project_root
        env["PYTHONPATH"] = (
            str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
        )
        # The child resolves ComfyUI and the models the same way the
        # server does, rather than being told and going stale.
        env["CHECKPOINTS_DIR"] = str(self._layout.checkpoints_dir)
        env["LORAS_DIR"] = str(self._layout.loras_dir)
        comfy = self._layout.comfy_dir
        if comfy:
            env["COMFY_DIR"] = str(comfy)
        return env

    # -- stop --------------------------------------------------------------

    def request_stop(self, pid: int) -> None:
        """SIGINT: the runtime notices at its next step boundary."""
        self._signal(pid, signal.SIGINT)

    def kill(self, pid: int) -> None:
        """SIGKILL, to the process group, when the grace period ran out."""
        self._signal(pid, signal.SIGKILL)

    def _signal(self, pid: int, sig: signal.Signals) -> None:
        match = self._cmdline_marker_match(pid)
        if match is False:
            logger.warning(
                "refusing to signal pid %s: not a graph execution child "
                "(pid reuse?)", pid,
            )
            return
        try:
            os.killpg(os.getpgid(pid), sig)
        except OSError:
            # The group is gone, or the pid is not ours to signal. Either
            # way there is nothing to deliver and nothing to report.
            try:
                os.kill(pid, sig)
            except OSError:
                logger.debug("pid %s already gone; %s not delivered", pid, sig.name)

    # -- liveness ---------------------------------------------------------

    def is_alive(self, pid: int) -> bool:
        with self._lock:
            proc = self._procs.get(pid)
        if proc is not None and proc.poll() is not None:
            return False  # reaped: exited, no zombie left behind
        match = self._cmdline_marker_match(pid)
        if match is None:
            # /proc unavailable: fall back to the bare probe, failing open
            # (a stuck run beats a spurious completion -- docs 07 F-12).
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        return match

    def _cmdline_marker_match(self, pid: int) -> bool | None:
        return cmdline_mentions(pid, CMDLINE_MARKER)

    # -- adoption ---------------------------------------------------------

    def find_running_all(self, execution_id) -> list[int]:
        """Live children claiming this execution, discovered by their argv.

        Discovered rather than stored, and that is the point. A pid kept
        in a row is a claim about the past: read it after a reboot and it
        names whatever process the number was recycled to, which is
        precisely the mistake the pid-reuse guard exists to prevent. The
        child's own argv says which execution it is serving, so this asks
        the live system instead of trusting a record.

        Every match, deliberately -- see the port's note on why this and
        `find_running` are different questions.
        """
        return find_by_argv(CMDLINE_MARKER, "--execution", str(execution_id))

    def find_running(self, execution_id) -> int | None:
        """The one child for this execution, or `None`.

        Returns ``None`` on a second match as well as on none. Two
        children claiming one execution is not a situation to guess about
        -- adopting one of them would replay one run's history into a row
        the other is still writing. Both are *still running*, which is why
        `find_running_all` exists alongside this: refusing to adopt is not
        the same as nothing being alive.
        """
        matches = self.find_running_all(execution_id)
        if len(matches) != 1:
            if matches:
                logger.error(
                    "graph execution %s: %d live children claim it (%s); "
                    "not adopting either",
                    execution_id, len(matches), matches,
                )
            return None
        return matches[0]

class InProcessGraphTaskGateway(GraphTaskGateway):
    """Runs the child *body* on a thread, in this process.

    Not "the old way round a new interface": it calls the very same
    ``run_execution`` the child module calls, with a ``threading.Event``
    where the child has a SIGINT handler. So the two differ in where the
    interpreter state lives and in nothing else -- same runtime, same
    event records, same outcome semantics.

    What it is for is the rollout. Until process isolation is the default,
    the supervisor, the watcher, the event format and the reconciliation
    all have to be exercisable on every run, and having only the child
    path exercised means the code that replaced it is untested until the
    moment it becomes the only path. Being able to select either means the
    flip is a one-line change in the composition root rather than a leap.

    It is also the rollback. If the child path turns out to break something
    on hardware nobody tested, the default goes back without a revert.
    """

    def __init__(self, graph_registry=None, runtime_factory=None) -> None:
        self._registry = graph_registry
        # Seam for tests that need to observe the run -- a counting
        # memory releaser, say. Takes the writer so the factory can wire
        # its runtime to the event file exactly as build_runtime does.
        self._runtime_factory = runtime_factory
        self._threads: dict[int, threading.Thread] = {}
        self._cancels: dict[int, threading.Event] = {}
        self._lock = threading.Lock()

    def spawn(self, launch: GraphTaskLaunch) -> int:
        from backend.domain.graph import GraphDefinition

        from .graph_event_stream import ExecutionEventWriter
        from .graph_task_worker import build_runtime, run_execution

        # A pid, because the port says so and because every caller above
        # (stop, reconcile, the API) speaks in pids. Negative and
        # distinguishable from any real one -- a real pid is never negative,
        # so nothing can mistake this for a process.
        pid = -next(self._next_id)
        cancel = threading.Event()
        writer = ExecutionEventWriter(Path(launch.event_path))

        def body() -> None:
            try:
                graph = GraphDefinition.from_dict(
                    json.loads(Path(launch.graph_path).read_text(encoding="utf-8"))
                )
                make = self._runtime_factory or (
                    lambda w: build_runtime(w, self._registry)
                )
                run_execution(graph, writer, cancel, make(writer))
            except Exception as exc:  # noqa: BLE001 -- the watcher needs to hear
                writer.outcome(error=f"{type(exc).__name__}: {exc}", results_count=0)
            finally:
                writer.close()

        thread = threading.Thread(
            target=body, name=f"backend-graph-inproc-{pid}", daemon=True
        )
        with self._lock:
            self._threads[pid] = thread
            self._cancels[pid] = cancel
        thread.start()
        return pid

    _next_id = iter(range(1, 1 << 30))

    def request_stop(self, pid: int) -> None:
        with self._lock:
            cancel = self._cancels.get(pid)
        if cancel is not None:
            cancel.set()

    def kill(self, pid: int) -> None:
        # There is nothing to escalate to: the runtime checks between
        # steps, and a thread cannot be killed from outside at all. Saying
        # so is better than pretending, and better than silently doing
        # nothing while the caller waits out a grace period.
        self.request_stop(pid)
        logger.warning(
            "in-process graph execution cannot be hard-killed; the cancel "
            "event was set again (pid %s)", pid,
        )

    def is_alive(self, pid: int) -> bool:
        with self._lock:
            thread = self._threads.get(pid)
        return thread is not None and thread.is_alive()

    def find_running_all(self, _execution_id) -> list[int]:
        """Always empty, for the reason `find_running` is always `None`.

        A thread cannot outlive the process it is in, so there is never a
        second one to find. ``[]`` rather than a one-element list is the
        honest answer to "is anything still running", and it is what makes
        the caller leave such a row alone only when something really is.
        """
        return []

    def find_running(self, _execution_id) -> int | None:
        """Always ``None``: a thread cannot outlive the process it is in.

        Not a limitation to apologise for -- it is the honest answer, and
        the one the reconciler's fallback path is built around. An
        in-process run dies with the server, so a restart genuinely is
        the end of it; there is nothing to look for.
        """
        return None

    def reap(self, pid: int) -> None:
        """Drop the finished thread and cancel event.

        Called by the supervisor once it has finalised a row. Without it
        the maps grow for the life of the server, one entry per execution
        ever run, which is the kind of leak that is invisible until it is
        large.
        """
        with self._lock:
            self._threads.pop(pid, None)
            self._cancels.pop(pid, None)
