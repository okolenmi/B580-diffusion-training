"""GraphTaskGateway port -- running a graph execution in a child process.

Same shape as ``DatasetTaskGateway``, and for the same reason: the child
needs a pid, and everything else about supervision (liveness, stop,
escalation, pid-reuse safety) is the same problem twice.

What is different from the dataset-task case is the reporting. A dataset
task writes progress straight into ``backend.db``, which WAL makes safe
for a second writer. A graph execution cannot: its results are domain
objects built by the runtime and its live monitor reports are in-memory
per process, so neither survives being reconstructed from a row. Hence
the event file (``infrastructure/graph_event_stream.py``) -- the child
appends, the parent tails.

``is_alive`` is PID-reuse aware for the reason it is on the dataset-task
port (docs 07 F-12, fixed in ``a1e0358``): ``kill(pid, 0)`` answers
"does *a* process hold this number", and after a reboot it will happily
answer yes about something else. The adapter therefore checks the child's
cmdline against a marker, via the one implementation of that check in
``infrastructure/process_identity.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ...domain.graph import GraphDefinition
from ...domain.value_objects import ExecutionId


class GraphLaunchError(Exception):
    """The child could not be started at all.

    Distinct from "the child started and the graph failed": the caller can
    repair the row itself in the first case, while the second is a normal
    run outcome the child reports.
    """


@dataclass(frozen=True, slots=True)
class GraphTaskLaunch:
    """Everything the child needs, as paths.

    The graph goes in a file rather than on argv for two reasons: it can
    be large (a saved graph with many nodes approaches ``ARG_MAX``), and
    passing one as a single argument means a graph containing a quote or
    a newline is a shell-quoting problem rather than a JSON one.

    ``event_path`` is the append-only report channel described in
    ``graph_event_stream``; the parent creates the directory and the
    child opens the file for append.

    The two memory numbers ride along as plain floats because they
    exist only on the execution row: the graph file carries the
    *graph's* settings, not this run's admission result (overrides,
    peak-derived demand). ``None`` means the number was never supplied;
    the child builds its GraphMemory with that number unknown rather
    than this dataclass inventing a zero for it.
    """

    execution_id: ExecutionId
    graph_path: Path
    event_path: Path
    log_path: Path
    #: Allocator MB this run may use (the fraction backstop's input).
    memory_budget_mb: float | None = None
    #: Device MB admission granted this run (the physical check's input).
    memory_grant_mb: float | None = None



class GraphTaskGateway(ABC):
    @abstractmethod
    def spawn(self, launch: GraphTaskLaunch) -> int:
        """Start the child; returns its pid.

        Raises ``GraphLaunchError`` when the process could not be started
        at all, so the caller can fail the row itself.
        """
        raise NotImplementedError

    @abstractmethod
    def request_stop(self, pid: int) -> None:
        """Ask the child to stop at its next step boundary.

        Cooperative, and deliberately not the same thing as ``kill``: the
        gateway's grace period exists so a stopping run can flush its event
        file and let ``release_memory`` run. A hard kill loses the tail of
        that file, which is exactly the torn line the reader is built to
        tolerate -- but not to prefer.
        """
        raise NotImplementedError

    @abstractmethod
    def kill(self, pid: int) -> None:
        """Force-kill the child's process group; fail-open on dead pids."""
        raise NotImplementedError

    @abstractmethod
    def is_alive(self, pid: int) -> bool:
        """True only for a live process that is one of our graph children."""
        raise NotImplementedError

    @abstractmethod
    def find_running_all(self, execution_id: ExecutionId) -> list[int]:
        """Every live child claiming ``execution_id``, ascending by pid.

        ``find_running`` deliberately collapses "two children claim it" to
        ``None`` rather than guessing which to adopt, and that is right for
        adopting. It is wrong for asking whether anything is still running,
        because the processes it collapsed are still running: answering
        ``None`` there lets a caller conclude the row is debris and fail it
        while two trainers hold the card. Ascending order, so a caller that
        does act on the list gets a deterministic answer.
        """
        raise NotImplementedError

    def find_running(self, execution_id: ExecutionId) -> int | None:
        """The pid of a still-running child for ``execution_id``, if any.

        Asked once per unfinished row at startup, and the whole reason the
        port has this method: a child outlives the server that started it
        (its own session, its own parent after a restart), so a restart is
        not by itself a reason to throw the run away.

        ``None`` for "no such child" and for "cannot tell" alike, because
        the caller's response to both is the same -- fail the row -- and
        claiming to know when ``/proc`` could not answer would be a lie
        with consequences.
        """
        raise NotImplementedError


__all__ = [
    "GraphDefinition",
    "GraphLaunchError",
    "GraphTaskGateway",
    "GraphTaskLaunch",
]