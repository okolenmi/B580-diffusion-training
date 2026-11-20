"""Commands from outside a run: a page asks a node to do something while the graph is running.

A command is ``{"id", "node", "action"}``. Whatever carries it to the run (the file inbox of ``backend/infrastructure/command_inbox``,
or a test) calls ``ActionDispatcher.dispatch`` from its own thread. The dispatcher builds the event from the node's declared
``ActionModule`` -- never from anything in the command -- and queues it on the run's hub, which delivers it at the next safe point on
the run thread. Every command is answered with ``node:<uid>`` frames ``{"kind": "action", "id", "action", "status", "detail"}``:
``queued`` now, then ``delivered``, ``rejected`` or ``failed``.
"""

from __future__ import annotations

import threading

from .events import EventHub, Reporter


class ActionDispatcher:
    """One per process; bound to each run's hub while that run is live."""

    def __init__(self, reporter: Reporter | None = None) -> None:
        self._reporter = reporter
        self._lock = threading.Lock()
        self._hub: EventHub | None = None
        self._classes: dict[str, type] = {}

    def bind(self, hub: EventHub, classes_by_uid: dict[str, type]) -> None:
        with self._lock:
            self._hub, self._classes = hub, dict(classes_by_uid)

    def unbind(self) -> None:
        with self._lock:
            self._hub, self._classes = None, {}

    def dispatch(self, command: dict) -> None:
        cid, uid, name = (str(command.get(k) or "") for k in ("id", "node", "action"))
        with self._lock:
            hub, cls = self._hub, self._classes.get(uid)
        if hub is None:
            self._answer(uid, cid, name, "rejected", "the run is not accepting commands")
            return
        action = next((a for a in getattr(cls, "ACTIONS", ()) if a.name == name), None) if cls is not None else None
        if action is None:
            self._answer(uid, cid, name, "rejected", f"node {uid!r} has no action {name!r}")
            return
        event = cls.EVENTS_IN[action.event].payload(**action.defaults)
        self._answer(uid, cid, name, "queued", "")
        hub.request(uid, action.event, event, lambda status, detail: self._answer(uid, cid, name, status, detail))

    def _answer(self, uid: str, cid: str, name: str, status: str, detail: str) -> None:
        if self._reporter is not None:
            self._reporter.report(f"node:{uid}", {"kind": "action", "id": cid, "action": name, "status": status, "detail": detail})
