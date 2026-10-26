"""The event plane: typed notifications between nodes during a run.

See rework/contracts/events.md. Dispatch is synchronous and happens on the run thread, so
an emitter that pauses at a safe point (the end of an epoch) knows every subscriber has
finished when ``emit`` returns.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Protocol
from collections.abc import Callable

MAX_DEPTH = 8


class EventError(RuntimeError):
    """Base class for hub failures; the runtime reports them as a failed run."""


class EventLoopError(EventError):
    pass


class EventHandlerError(EventError):
    pass


class EventDeliveryError(EventError):
    pass


class EventThreadError(EventError):
    pass


def _json_safe(value: Any, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth > 2:
        return repr(value)[:80]
    if isinstance(value, (list, tuple)):
        return [_json_safe(v, depth + 1) for v in list(value)[:20]]
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in list(value.items())[:200 if depth == 0 else 20]}  # a step's metrics are a long flat dict
    return repr(value)[:80]


@dataclass(frozen=True)
class EventType:
    """Base of every event payload. Subclass as a frozen dataclass; keep payloads small:
    heavy data travels in shared handles and the payload carries an index or version."""

    always_delivered = False

    def summary(self) -> dict:
        if not is_dataclass(self):
            return {}
        return {f.name: _json_safe(getattr(self, f.name)) for f in fields(self)}


@dataclass(frozen=True)
class RunStarted(EventType):
    pass


@dataclass(frozen=True)
class RunStopped(EventType):
    always_delivered = True
    reason: str = "stopped"


@dataclass(frozen=True)
class StepFinished(EventType):
    step: int
    total: int
    metrics: dict


@dataclass(frozen=True)
class EpochEnding(EventType):
    epoch: int
    samples: int


@dataclass(frozen=True)
class EpochStarted(EventType):
    epoch: int


@dataclass(frozen=True)
class PromptsChanged(EventType):
    version: int
    changed: int
    of: int


@dataclass(frozen=True)
class CheckpointDue(EventType):
    step: int
    reason: str


@dataclass(frozen=True)
class StopRequested(EventType):
    reason: str


@dataclass(frozen=True)
class MemoryPressure(EventType):
    level: str


class Reporter(Protocol):
    """Where frames for the UI go. The server supplies the real one."""

    def report(self, stream: str, frame: dict) -> None: ...


@dataclass(frozen=True)
class Wire:
    source: str
    port: str
    target: str
    handler_port: str


@dataclass
class _Subscriber:
    wire: Wire
    handler_name: str


class EventContext:
    """What a handler gets besides the event: a way to emit from its own node."""

    def __init__(self, hub: EventHub, uid: str):
        self._hub = hub
        self.uid = uid

    def emit(self, port: str, event: EventType) -> None:
        self._hub.emit(self.uid, port, event)

    def post(self, port: str, event: EventType) -> None:
        self._hub.post(self.uid, port, event)

    def should_cancel(self) -> bool:
        return self._hub.stopped


class EventHub:
    """One per run. Created on the run thread; ``emit`` must be called from it."""

    def __init__(self, reporter: Reporter | None = None, *, ranks: dict[str, int] | None = None,
                 max_depth: int = MAX_DEPTH, clock: Callable[[], float] = time.monotonic):
        self._reporter = reporter
        self._ranks = ranks or {}
        self._max_depth = max_depth
        self._clock = clock
        self._t0 = clock()
        self._seq = 0
        self._thread = threading.get_ident()
        self._subs: dict[tuple[str, str], list[_Subscriber]] = {}
        self._nodes: dict[str, Any] = {}
        self._posted: queue.SimpleQueue = queue.SimpleQueue()
        self._stack: list[Wire] = []
        self._on_stop: Callable[[StopRequested], None] | None = None
        self.stopped = False

    def connect(self, wire: Wire, handler_name: str) -> None:
        subs = self._subs.setdefault((wire.source, wire.port), [])
        subs.append(_Subscriber(wire, handler_name))
        subs.sort(key=lambda s: (self._ranks.get(s.wire.target, 0), s.wire.target, s.wire.handler_port))

    def attach(self, uid: str, node: Any) -> None:
        self._nodes[uid] = node

    def set_stop_callback(self, callback: Callable[[StopRequested], None]) -> None:
        self._on_stop = callback

    def mark_stopped(self) -> None:
        self.stopped = True

    def emit(self, uid: str, port: str, event: EventType) -> None:
        if threading.get_ident() != self._thread:
            raise EventThreadError(f"emit from {uid}.{port} on a worker thread; use post()")
        if self.stopped and not event.always_delivered:
            return
        subs = self._subs.get((uid, port))
        if not subs and self._reporter is None and not isinstance(event, StopRequested):
            return
        self._mirror(uid, port, event)
        if subs:
            for sub in list(subs):
                self._deliver(sub, event)
        if isinstance(event, StopRequested) and self._on_stop is not None:
            self._on_stop(event)

    def post(self, uid: str, port: str, event: EventType) -> None:
        self._posted.put((uid, port, event))

    def drain(self) -> int:
        """Deliver events posted from other threads. Call from the run thread at a safe point."""
        count = 0
        while True:
            try:
                uid, port, event = self._posted.get_nowait()
            except queue.Empty:
                return count
            self.emit(uid, port, event)
            count += 1

    def _deliver(self, sub: _Subscriber, event: EventType) -> None:
        wire = sub.wire
        if wire in self._stack:
            raise EventLoopError(f"event loop through {wire.source}.{wire.port} -> {wire.target}.{wire.handler_port}")
        if len(self._stack) >= self._max_depth:
            raise EventLoopError(f"event nesting deeper than {self._max_depth} at {wire.source}.{wire.port}")
        node = self._nodes.get(wire.target)
        if node is None:
            raise EventDeliveryError(
                f"{wire.source}.{wire.port} emitted before {wire.target} was built (wire to {wire.handler_port})")
        handler = getattr(node, sub.handler_name)
        self._stack.append(wire)
        try:
            handler(event, EventContext(self, wire.target))
        except EventError:
            raise
        except Exception as exc:  # noqa: BLE001 -- attribute the failure to the node
            raise EventHandlerError(
                f"{wire.target}.{wire.handler_port} failed on {type(event).__name__}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            self._stack.pop()

    def _mirror(self, uid: str, port: str, event: EventType) -> None:
        if self._reporter is None:
            return
        self._seq += 1
        self._reporter.report(f"node:{uid}", {
            "kind": "event", "name": port, "source": uid, "type": type(event).__name__,
            "summary": event.summary(), "seq": self._seq, "t": round(self._clock() - self._t0, 4),
        })
