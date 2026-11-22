"""Stopping a child process: escalation of an ignored stop, and giving up on a run that can no longer be watched."""

from __future__ import annotations

import logging
import threading

from ..domain.value_objects import ExecutionId
from .ports.graph_task_gateway import GraphTaskGateway


logger = logging.getLogger(__name__)


class ChildStopper:
    """Owns the two decisions about a child that must stop: when a cooperative stop has been ignored long enough to kill it, and what to do with a run whose watcher failed."""

    def __init__(self, *, gateway: GraphTaskGateway, grace: float) -> None:
        self._gateway = gateway
        # Long enough for a stopping run to flush its event file and release device memory, short enough that a wedged
        # run does not hold the card forever.
        self._stop_grace = grace

    def escalate(self, execution_id: ExecutionId, pid: int) -> None:
        """Hard-kill a run that ignored a cooperative stop."""
        threading.Event().wait(self._stop_grace)
        if not self._gateway.is_alive(pid):
            return
        logger.warning(
            "graph execution %s (pid %s) ignored the stop for %.0fs; killing",
            execution_id, pid, self._stop_grace,
        )
        self._gateway.kill(pid)

    def abandon(self, execution_id: ExecutionId, pid: int) -> bool:
        """Stop a run we have lost the ability to watch. True if it was live.

        This exists for one invariant: **a row is never terminal while its
        child is alive.** A terminal row releases the single-active check,
        so writing one for a run that is still holding the card is how two
        trainers end up on one 12 GB device -- and an unwatched run is also
        a run the API can no longer stop, because ``cancel`` needs the pid
        this watcher was holding.

        So when the watcher gives up, it does not simply write the row off.
        It asks the child to stop, escalates to a kill if that is ignored,
        and only then lets the row be finalised -- with a note that says
        supervision failed rather than claiming the process died.

        Returns ``False`` when the child had already gone, which is the
        ordinary case: a watcher that breaks because liveness said no has
        nothing to abandon.
        """
        try:
            if not self._gateway.is_alive(pid):
                return False
        except Exception:  # noqa: BLE001 -- liveness is what failed
            # Assume the worst. The alternative is leaving a process that
            # holds the card unaccounted for, on the strength of a check
            # that did not answer.
            logger.warning(
                "could not determine whether pid %s is alive; treating it as "
                "alive and stopping it", pid,
            )

        logger.error(
            "graph execution %s can no longer be supervised; stopping its "
            "child (pid %s) rather than leaving it running unobserved",
            execution_id, pid,
        )
        try:
            self._gateway.request_stop(pid)
        except Exception:  # noqa: BLE001 -- escalate regardless
            logger.exception("asking pid %s to stop failed", pid)
        threading.Event().wait(self._stop_grace)
        try:
            still_running = self._gateway.is_alive(pid)
        except Exception:  # noqa: BLE001 -- liveness is what failed
            # Liveness has already failed once on this run, so a second
            # failure here is expected rather than surprising. It must not
            # be read as "so it probably stopped": not knowing is the one
            # state in which the card is still held and the run is still
            # unstoppable through the API. Unknown means alive.
            logger.warning(
                "cannot tell whether pid %s stopped; killing it rather than "
                "leaving it unaccounted for", pid,
            )
            still_running = True
        if still_running:
            logger.warning(
                "pid %s did not stop within %.0fs of supervision failing; "
                "killing", pid, self._stop_grace,
            )
            try:
                self._gateway.kill(pid)
            except Exception:  # noqa: BLE001 -- nothing further to try
                logger.exception("killing pid %s failed", pid)
        return True
