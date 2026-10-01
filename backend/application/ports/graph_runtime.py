"""GraphRuntime port -- authoritative validation + execution of graphs.

The executor contract behind M4 (``docs/design/backend/05-graph-runtime.md``):

* ``validate`` returns a *complete* structured issue list (never raises,
  never stops at the first problem) -- the same checks ``execute``
  performs defensively, computed with real Python types (``issubclass``,
  ``Port.choices``, declared param types), so the frontend's local type
  check is UX sugar and this is the truth;
* ``execute`` runs the graph synchronously in the caller's thread (the
  application supervisor owns threads and cancellation), reports each
  finished node through ``on_node_done`` (partial results + progress
  events), and returns outcomes instead of raising -- a failing node is
  a normal result to report;
* ``release_memory`` returns freed device memory to the driver after a
  run (gc first, then the caching allocator -- same reason as the legacy
  worker's ``finally`` block).

Issue/result dataclasses cross this boundary constantly, so they live
here; ``NodeResult`` itself lives in ``domain/graph`` because the
execution entity stores it (domain must not import application).
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from ...domain.graph import GraphDefinition, NodeResult

class IssueSeverity(str, Enum):
    """How much a validation finding matters.

    ``ERROR`` blocks a run (the graph would misbehave); ``WARNING`` is
    advice the submitter may ignore. ``str``-valued, so the wire shape is
    unchanged and the editor's ``issue.severity === "error"`` keeps
    working (docs 08 S-24).
    """

    ERROR = "error"
    WARNING = "warning"

    @property
    def blocks(self) -> bool:
        return self is IssueSeverity.ERROR


@dataclass(frozen=True, slots=True)
class GraphIssue:
    """One validation finding; ``severity.blocks`` says whether it stops
    the run.

    ``node_id``/``edge_index``/``param`` localize the finding when they
    apply (each is None otherwise); ``message`` is human-readable and
    always populated. The stable machine-readable part is ``code`` --
    see doc 05 section 4 for the full table.
    """

    severity: IssueSeverity | str
    code: str
    message: str
    node_id: str | None = None
    edge_index: int | None = None
    param: str | None = None


def issue_to_dict(issue: GraphIssue) -> dict:
    """JSON-safe form of an issue (one converter for the API's validate
    response *and* the run-rejection ``details`` -- same shape twice)."""
    return {
        "severity": str(issue.severity),
        "code": issue.code,
        "message": issue.message,
        "node_id": issue.node_id,
        "edge_index": issue.edge_index,
        "param": issue.param,
    }


@dataclass(frozen=True, slots=True)
class GraphOutcome:
    """Result of ``execute``: every node that ran, plus a graph-level
    error message when the run did not complete cleanly.

    ``error`` set means the execution ends as ``error`` (validation
    defense, cycle, or a node's build() raising -- the failed node is
    also in ``results`` with ``ok=False``). ``error=None`` + a set
    cancel event means ``stopped``; both clean means ``finished``.
    """

    results: tuple[NodeResult, ...] = ()
    error: str | None = None


NodeDoneCallback = Callable[[NodeResult], None]
"""Called after every node that completes (success or failure), in
execution order. Exceptions raised by the callback are logged and
swallowed by the implementation -- persisting progress must never abort
the run itself (the final write carries the authoritative results)."""


class GraphRuntime(ABC):
    """Validate + execute one graph against the real node registry."""

    @abstractmethod
    def validate(self, graph: GraphDefinition) -> tuple[GraphIssue, ...]:
        """Every finding, deterministic order (submission order of
        nodes, then edges by index; params in INPUTS order)."""
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        graph: GraphDefinition,
        *,
        cancel_event: threading.Event,
        on_node_done: NodeDoneCallback | None = None,
    ) -> GraphOutcome:
        """Run the graph in this thread until done, failed, or cancelled.

        ``cancel_event`` is cooperative twice over: checked between
        nodes by this executor, and passed to every node inside the
        ``ExecutionContext`` (nodes poll it between heavy steps, never
        mid-backward-pass). Never raises for graph/node problems --
        they come back in the outcome.
        """
        raise NotImplementedError

    @abstractmethod
    def release_memory(self) -> None:
        """Best-effort return of freed device memory (no-op-safe).

        ``execute`` calls this itself in its own ``finally`` -- the
        runtime owns the device state it allocated, so a caller cannot
        forget the cleanup (docs 08 S-05). It stays on the port for
        callers that allocate outside a run.
        """
        raise NotImplementedError
