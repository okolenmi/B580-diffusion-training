"""SaveGraph -- upsert one named graph into the library.

Light shape check only: the payload must carry a ``nodes`` list (and
``edges`` if present) -- enough that we never store obvious garbage,
not a full validation. Class names, edges, and params are checked at
*run* time by ``StartGraphExecution``; saving must keep working for a
graph that references a node class which doesn't exist *yet* (forward
compatibility, see the port's docstring).

``created`` (first save vs replace) lets the API answer 201 vs 200.
"""

from __future__ import annotations

from ..dto import SaveGraphResult, to_saved_graph_dto
from ..errors import InvalidQueryError
from ..ports.graph_library import GraphLibrary, normalize_graph_name
from ...domain.graph import GRAPH_FORMAT

MAX_DESCRIPTION = 1000


class SaveGraph:
    def __init__(self, library: GraphLibrary) -> None:
        self._library = library

    def execute(
        self, name: str, graph: dict, *, description: str = ""
    ) -> SaveGraphResult:
        try:
            clean_name = normalize_graph_name(name)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        if len(description) > MAX_DESCRIPTION:
            raise InvalidQueryError(
                f"description must be at most {MAX_DESCRIPTION} characters"
            )
        payload = self._payload(graph)
        saved, created = self._library.save(
            clean_name, payload, description=description
        )
        return SaveGraphResult(graph=to_saved_graph_dto(saved), created=created)

    @staticmethod
    def _payload(graph: dict) -> dict:
        """Stamp the format version, keep everything else verbatim."""
        if not isinstance(graph, dict):
            raise InvalidQueryError("graph must be an object")
        nodes = graph.get("nodes")
        if not isinstance(nodes, list):
            raise InvalidQueryError("graph.nodes must be a list")
        edges = graph.get("edges")
        if edges is None:
            edges = []
        if not isinstance(edges, list):
            raise InvalidQueryError("graph.edges must be a list")
        payload = dict(graph)
        payload["nodes"] = nodes
        payload["edges"] = edges
        stamp = payload.get("format")
        if not isinstance(stamp, int) or isinstance(stamp, bool):
            payload["format"] = GRAPH_FORMAT
        return payload
