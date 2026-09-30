"""DeleteGraph -- remove one saved graph (404 when unknown)."""

from __future__ import annotations

from ..dto import DeleteGraphResult
from ..errors import GraphNotFoundError, InvalidQueryError
from ..ports.graph_library import GraphLibrary, normalize_graph_name


class DeleteGraph:
    def __init__(self, library: GraphLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> DeleteGraphResult:
        try:
            clean_name = normalize_graph_name(name)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        if not self._library.delete(clean_name):
            raise GraphNotFoundError(f"no saved graph named {clean_name!r}")
        return DeleteGraphResult(deleted=True)
