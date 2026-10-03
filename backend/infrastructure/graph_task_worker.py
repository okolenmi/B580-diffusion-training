"""The child process: runs one graph execution and reports through a file.

Launched as ``python -m backend.infrastructure.graph_task_worker
--execution N --graph FILE --events FILE``. Nothing here is imported by
the server, and the server imports nothing from here -- the two agree only
on the event file's format, which is what ``graph_event_stream`` owns.

Why it exists (WP-22): graph execution used to run in a thread inside the
API server, so a device fault, an OOM kill or a driver-level reset took the
server and the run down together, and a long node build competed with
everything else in the process. Out here, the worst outcome is that *this*
process dies, and the row says so.

Three things it deliberately does not do
----------------------------------------
* **No database access.** It writes no rows. The server owns the execution
  row and every status change on it, which keeps the compare-and-swap that
  the whole lifecycle rests on in exactly one place.
* **No validation.** ``validate`` is cheap, pure, and needed by the editor
  on every keystroke, so it stays in the server. This child runs a graph
  the server already validated.
* **No second source of truth for live data.** The monitor bus is a
  per-process in-memory deque with per-subscriber asyncio queues, neither
  of which a child can reach -- so reports are appended to the event file
  and the server republishes them onto its own bus. The file is the
  history, which is what makes "open the dashboard mid-run and see
  everything so far" still true.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import traceback
from pathlib import Path

from ..python_floor import require_python


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backend.infrastructure.graph_task_worker",
        description="Run one graph execution and report through an event file.",
    )
    p.add_argument("--execution", type=int, required=True)
    p.add_argument("--graph", required=True, help="JSON GraphDefinition")
    p.add_argument("--events", required=True, help="append-only event file")
    return p


class _EventMonitorBus:
    """Duck-typed MonitorBus that appends instead of dispatching.

    ``nodes/`` only requires ``report``/``clear`` of whatever object
    ``ExecutionContext`` carries -- that duck-typing is what lets the same
    runtime work in both processes. Sanitizing happens in the writer, so
    this does not repeat it.
    """

    def __init__(self, writer) -> None:
        self._writer = writer

    def report(self, monitor_id: str, data: dict) -> None:
        self._writer.monitor(monitor_id, data)

    def clear(self, monitor_id: str) -> None:
        # "Clear" means "stop replaying this history". Across a process
        # boundary the history is the file, and a monitor the run has
        # stopped reporting simply produces nothing more -- there is
        # nothing to clear, and truncating the shared file would delete
        # every other monitor's records.
        pass


def build_runtime(writer, registry=None):
    """The one runtime both gateways run: the real one, wired to the file.

    Takes the registry so the in-process gateway can reuse the server's
    already-discovered one instead of paying discovery again; the child
    passes nothing and discovers its own.
    """
    from .graph.discovery import NodeRegistry
    from .graph.runtime import ReflectedGraphRuntime

    return ReflectedGraphRuntime(
        registry if registry is not None else NodeRegistry(),
        monitor_bus=_EventMonitorBus(writer),
    )


def run_execution(graph, writer, cancel: threading.Event, runtime):
    """Run one graph, reporting every record through ``writer``.

    The single producer, shared by both gateways. A child process builds a
    runtime with ``build_runtime`` and calls this from ``main``; the
    in-process gateway builds one on a thread with a ``threading.Event``
    where the child has a SIGINT handler. Identical behaviour either way
    is the point: the alternative is two implementations of "run a graph
    and report it", and they would not stay identical -- and the whole
    reason the event file exists is so the parent's side never has to care
    which one ran.

    Returns the runtime's outcome. A failing *graph* is a normal outcome,
    recorded and returned; a failing *harness* raises, so the caller
    decides whether that deserves an error exit.
    """
    outcome = runtime.execute(
        graph,
        cancel_event=cancel,
        on_node_done=lambda result: writer.node({
            "node_id": result.node_id,
            "ok": result.ok,
            "outputs": result.outputs,
            "error": result.error,
            "duration_ms": result.duration_ms,
        }),
    )
    writer.outcome(error=outcome.error, results_count=len(outcome.results))
    return outcome


def main(argv: list[str] | None = None) -> int:
    # The floor first, before the signal handlers: a child that cannot run
    # should say why on stderr, where the supervisor's log will keep it.
    require_python()

    # Signals first, before anything that can take time -- which is the
    # whole point, because a stop that arrives before this line lands on
    # Python's default handler and kills the process instead of asking it.
    # Measured at 37 ms of imports and argument parsing before the handlers
    # used to be installed: small enough that nobody hits it by hand, and
    # that is the only reason it survived. `cancel` and `signal` are
    # stdlib and already imported at module scope, so this costs nothing
    # and needs no project code.
    cancel = threading.Event()

    def _on_stop(_signum, _frame):
        # The gateway sends SIGINT for a cooperative stop and escalates to
        # SIGKILL after its grace period, so the only thing to do here is
        # let the runtime notice between steps.
        #
        # SIGTERM gets the same treatment, and used to get none: nothing
        # handled it, so the default disposition killed the child on
        # arrival. Measured on a 60000-node graph, mid-run:
        #
        #   SIGINT   exit 0,  outcome record written, 200 node records
        #   SIGTERM  exit -15, no outcome record,        86 node records
        #
        # A run stopped by SIGTERM is stopped *on purpose* -- `kill`,
        # `docker stop` and systemd all send it -- and with no record the
        # supervisor falls back to "execution process exited without
        # reporting an outcome (crashed, or a device fault killed it)",
        # which is false, sends the reader to a crash that did not happen,
        # and loses the results the run had already produced.
        #
        # Making it cooperative does mean a SIGTERM no longer kills
        # instantly: the stop is noticed at the next step boundary, so a
        # container runtime that wanted it gone sooner still sends
        # SIGKILL, and the project's own escalation is SIGINT -> SIGKILL
        # either way. What it buys is the outcome record and a released
        # device instead of an unexplained death.
        cancel.set()

    signal.signal(signal.SIGINT, _on_stop)
    signal.signal(signal.SIGTERM, _on_stop)

    args = _build_parser().parse_args(argv)

    # Before anything imports torch: SYCL reads these at its own runtime
    # init, so setting them even slightly late is too late.
    from nodes.xpu_env import set_xpu_perf_env_vars

    set_xpu_perf_env_vars()

    from backend.domain.graph import GraphDefinition

    from .graph_event_stream import ExecutionEventWriter

    writer = ExecutionEventWriter(Path(args.events))

    try:
        graph = GraphDefinition.from_dict(
            json.loads(Path(args.graph).read_text(encoding="utf-8"))
        )
        outcome = run_execution(graph, writer, cancel, build_runtime(writer))
        return 1 if outcome.error else 0
    except Exception as exc:  # noqa: BLE001 -- the server needs to hear about this
        writer.outcome(error=f"{type(exc).__name__}: {exc}", results_count=0)
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        writer.close()


if __name__ == "__main__":
    sys.exit(main())
