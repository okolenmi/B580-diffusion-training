"""ContractGraphRuntime -- the ``GraphRuntime`` port over ``next/graph``.

Validation and execution are the node tree's own (``GraphRunner``); this class only translates: the document
in, the port's issue and result types out. A failing node or an invalid graph never raises -- it comes back as an
outcome, as the port promises.

The memory release after a run is injected so tests never touch the device. Skipped nodes (after a stop or a
failure upstream) are left out of the reported results: the row lists what ran.
"""

from __future__ import annotations

import gc
import logging
import threading
from collections.abc import Callable
from typing import Any

from ...application.ports.graph_runtime import (
    GraphIssue,
    GraphOutcome,
    GraphRuntime,
    IssueSeverity,
    NodeDoneCallback,
)
from ...domain.graph import GraphDefinition, NodeResult
from . import tree

logger = logging.getLogger(__name__)


def default_memory_releaser() -> None:
    """gc first (cycles the refcount cannot free), then the device caching allocator, which is what actually
    returns memory to the driver. Imported lazily: the device module pulls in torch, which must happen on the first
    run, never at startup."""
    gc.collect()
    try:
        from nodes.components.device import DeviceContext  # noqa: PLC0415

        DeviceContext.for_device("xpu").empty_cache()
    except Exception as exc:  # noqa: BLE001 -- best-effort, never masks a result
        logger.info("xpu cache release unavailable (non-fatal): %s", exc)


def to_issue(issue: Any) -> GraphIssue:
    return GraphIssue(
        severity=IssueSeverity(issue.severity),
        code=issue.code,
        message=issue.message,
        node_id=issue.uid,
        edge_index=issue.edge_index,
        param=issue.port,
        plane=issue.plane,
    )


def to_result(result: Any) -> NodeResult:
    return NodeResult(
        node_id=result.uid,
        ok=result.ok,
        outputs=dict(result.outputs),
        error=result.error,
        duration_ms=result.duration_ms,
    )


class _NodeDoneObserver:
    """Hands each finished node to the caller; a skipped one never arrives here."""

    def __init__(self, callback: NodeDoneCallback | None) -> None:
        self._callback = callback

    def node_state(self, uid: str, state: str, detail: dict) -> None:
        pass

    def node_done(self, result: Any) -> None:
        if self._callback is not None:
            self._callback(to_result(result))

    def run(self, state: str, error: str | None) -> None:
        pass


class ContractGraphRuntime(GraphRuntime):
    def __init__(
        self,
        registry: Any | None = None,
        *,
        reporter: Any | None = None,
        env: Any | Callable[[], Any] | None = None,
        releaser: Callable[[], None] = default_memory_releaser,
        runtime_factory: Callable[[Any], Any] | None = None,
        overrides: dict | None = None,
    ) -> None:
        self._registry = registry or tree.NodeRegistry.default()
        self._reporter = reporter
        self._env = env
        self._releaser = releaser
        self._runtime_factory = runtime_factory
        # Settings the server decided for this run (the admitted memory budget): they win over the graph's own.
        self._overrides = dict(overrides or {})

    def _runner(self, observer: Any | None = None) -> Any:
        env = self._env() if callable(self._env) else self._env
        return tree.GraphRunner(
            self._registry, reporter=self._reporter, observer=observer, env=env, releaser=self._releaser,
            runtime_factory=self._runtime_factory,
        )

    def validate(self, graph: GraphDefinition) -> tuple[GraphIssue, ...]:
        return tuple(to_issue(i) for i in self._runner().validate(graph.doc).issues)

    def execute(
        self,
        graph: GraphDefinition,
        *,
        cancel_event: threading.Event,
        on_node_done: NodeDoneCallback | None = None,
    ) -> GraphOutcome:
        if self._overrides:
            graph = graph.with_settings({**graph.doc.settings, **self._overrides})
        outcome = self._runner(_NodeDoneObserver(on_node_done)).run(graph.doc, cancel_event)
        ran = tuple(to_result(r) for r in outcome.results if r.state != "skipped")
        return GraphOutcome(results=ran, error=outcome.error)

    def release_memory(self) -> None:
        self._releaser()
