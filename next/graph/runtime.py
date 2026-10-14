"""Running a validated graph (rework/contracts/runs.md)."""

from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .document import GraphDoc
from .events import EventError, EventHub, RunStopped, Reporter, StopRequested, Wire
from .node import COMPLETE, Completion, Environment, ExecutionContext, Node
from .registry import NodeRegistry
from .settings import SettingsRegistry
from .validate import Validation, validate

logger = logging.getLogger(__name__)

RUN_STREAM = "graph:run"


@dataclass(frozen=True)
class NodeResult:
    uid: str
    ok: bool
    outputs: dict
    error: str | None
    duration_ms: float
    state: str


@dataclass(frozen=True)
class RunOutcome:
    results: tuple[NodeResult, ...]
    error: str | None
    stopped: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class RunObserver(Protocol):
    def node_state(self, uid: str, state: str, detail: dict) -> None: ...
    def node_done(self, result: NodeResult) -> None: ...
    def run(self, state: str, error: str | None) -> None: ...


def describe(value: Any, depth: int = 0) -> Any:
    """JSON-safe view of an output. The real object only ever feeds the next node."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth < 3 and isinstance(value, (list, tuple)):
        return [describe(v, depth + 1) for v in list(value)[:20]]
    if depth < 3 and isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in list(value.items())[:20]}
    return {"_type": type(value).__name__, "_repr": repr(value)[:200]}


def _default_releaser() -> None:
    gc.collect()


@dataclass
class GraphRunner:
    registry: NodeRegistry
    settings_registry: SettingsRegistry | None = None
    reporter: Reporter | None = None
    observer: RunObserver | None = None
    app_defaults: dict = field(default_factory=dict)
    releaser: Callable[[], None] = _default_releaser
    runtime_factory: Callable[[ExecutionContext], Any] | None = None
    env: Environment | None = None

    def validate(self, doc: GraphDoc) -> Validation:
        return validate(doc, self.registry, self._settings(), self.env)

    def _settings(self) -> SettingsRegistry:
        if self.settings_registry is None:
            self.settings_registry = self.registry.settings_registry()
        return self.settings_registry

    def run(self, doc: GraphDoc, cancel_event: threading.Event | None = None) -> RunOutcome:
        cancel = cancel_event or threading.Event()
        doc = doc.for_run()
        validation = self.validate(doc)
        if not validation.ok:
            first = validation.errors[0]
            extra = f" (+{len(validation.errors) - 1} more)" if len(validation.errors) > 1 else ""
            return RunOutcome((), f"graph invalid: {first.message}{extra}")
        classes, _ = self.registry.load()
        order = validation.order or []
        t0 = time.monotonic()
        settings = self._settings().snapshot(doc.settings, self.app_defaults)
        hub = EventHub(self.reporter, ranks={uid: i for i, uid in enumerate(order)})
        hub.set_stop_callback(lambda ev: cancel.set())
        self._wire_events(doc, classes, hub)
        context = ExecutionContext(cancel_event=cancel, reporter=self.reporter, settings=settings, events=hub, env=self.env)
        if self.runtime_factory is not None:
            context.runtime = self.runtime_factory(context)

        outputs_by: dict[str, dict] = {}
        completion_by: dict[str, Completion] = {}
        retained: dict[str, Node] = {}
        results: list[NodeResult] = []
        error: str | None = None
        self._run_frame("run_started", None, t0)
        try:
            for uid in order:
                spec = doc.node(uid)
                cls = classes[spec.class_name]
                edges_in = doc.edges_to(uid)
                upstream = {e.to_port: completion_by.get(e.from_node, COMPLETE) for e in edges_in}
                if cancel.is_set():
                    hub.mark_stopped()
                    if not self._runs_after_stop(cls, edges_in, upstream, outputs_by):
                        self._state(uid, "skipped", t0, {})
                        results.append(NodeResult(uid, True, {}, None, 0.0, "skipped"))
                        continue
                inputs = self._gather(doc, uid, validation, outputs_by)
                node_context = context.scoped(uid, cls.READS_SETTINGS, cls.__name__, upstream)
                self._state(uid, "running", t0, {})
                started = time.monotonic()
                try:
                    node = cls(node_context)
                    outputs = node.build(**inputs)
                    self._check_outputs(uid, validation, outputs)
                    if cls.EVENTS_IN:
                        retained[uid] = node
                        hub.attach(uid, node)
                    hub.drain()
                except Exception as exc:  # noqa: BLE001 -- a failing node is a normal outcome
                    message = f"{type(exc).__name__}: {exc}"
                    result = NodeResult(uid, False, {}, message, (time.monotonic() - started) * 1000, "failed")
                    results.append(result)
                    self._state(uid, "failed", t0, {"error": message})
                    self._done(result)
                    error = message
                    break
                duration = (time.monotonic() - started) * 1000
                outputs_by[uid] = outputs
                completion_by[uid] = (node_context.reported_completion
                                      or node_context.any_upstream_unfinished() or COMPLETE)
                result = NodeResult(uid, True, {k: describe(v) for k, v in outputs.items()}, None, duration, "done")
                results.append(result)
                self._state(uid, "done", t0, {"duration_ms": round(duration, 1)})
                self._done(result)
            stopped = cancel.is_set()
            if stopped and error is None:
                hub.mark_stopped()
                try:
                    hub.emit("run", "run_stopped", RunStopped())
                except EventError as exc:
                    error = str(exc)
            final = "run_failed" if error else ("run_stopped" if stopped else "run_finished")
            self._run_frame(final, error, t0)
            return RunOutcome(tuple(results), error, stopped)
        finally:
            for uid in reversed(list(retained)):
                try:
                    retained[uid].close()
                except Exception:  # noqa: BLE001 -- cleanup must not lose results
                    logger.exception("closing node %s failed", uid)
            try:
                self.releaser()
            except Exception:  # noqa: BLE001
                logger.exception("memory release after a run failed")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _wire_events(doc: GraphDoc, classes: dict, hub: EventHub) -> None:
        for edge in doc.event_edges:
            target_cls = classes[doc.node(edge.to_node).class_name]
            handler = target_cls.EVENTS_IN[edge.to_port].handler
            hub.connect(Wire(edge.from_node, edge.from_port, edge.to_node, edge.to_port), handler)

    @staticmethod
    def _gather(doc: GraphDoc, uid: str, validation: Validation, outputs_by: dict) -> dict:
        spec = doc.node(uid)
        resolution = validation.resolutions[uid]
        inputs = dict(spec.params)
        wired: dict[str, list] = {}
        for edge in doc.edges_to(uid):
            wired.setdefault(edge.to_port, []).append(outputs_by[edge.from_node][edge.from_port])
        for name, values in wired.items():
            inputs[name] = values if resolution.inputs[name].multi else values[0]
        for name, port in resolution.inputs.items():
            if name in inputs:
                continue
            if port.multi:
                inputs[name] = []
            elif port.default is not None:
                inputs[name] = port.default
        return inputs

    @staticmethod
    def _check_outputs(uid: str, validation: Validation, outputs: dict) -> None:
        expected = set(validation.resolutions[uid].outputs)
        missing = expected - set(outputs)
        if missing:
            raise ValueError(f"build() did not produce declared output(s) {sorted(missing)}")

    @staticmethod
    def _runs_after_stop(cls: type[Node], edges_in, upstream: dict, outputs_by: dict) -> bool:
        if not cls.ACCEPTS_UNFINISHED or not any(c.unfinished for c in upstream.values()):
            return False
        return all(e.from_node in outputs_by for e in edges_in)

    def _state(self, uid: str, state: str, t0: float, detail: dict) -> None:
        frame = {"kind": "node_state", "uid": uid, "state": state, "t": round(time.monotonic() - t0, 4),
                 "detail": detail}
        if self.reporter is not None:
            self.reporter.report(RUN_STREAM, frame)
        if self.observer is not None:
            try:
                self.observer.node_state(uid, state, detail)
            except Exception:  # noqa: BLE001 -- observers never abort a run
                logger.exception("observer.node_state failed")

    def _done(self, result: NodeResult) -> None:
        if self.observer is not None:
            try:
                self.observer.node_done(result)
            except Exception:  # noqa: BLE001
                logger.exception("observer.node_done failed")

    def _run_frame(self, state: str, error: str | None, t0: float) -> None:
        frame = {"kind": "run", "state": state, "t": round(time.monotonic() - t0, 4), "error": error}
        if self.reporter is not None:
            self.reporter.report(RUN_STREAM, frame)
        if self.observer is not None:
            try:
                self.observer.run(state, error)
            except Exception:  # noqa: BLE001
                logger.exception("observer.run failed")
