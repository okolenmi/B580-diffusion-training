"""GetGraph -- one saved graph's stored payload, verbatim.

404 ``graph_not_found`` when the name is unknown; the payload is never
re-validated here (validation belongs to running the graph).
"""

from __future__ import annotations

from ..dto import SavedGraphDTO, to_saved_graph_dto
from ..errors import GraphNotFoundError, InvalidQueryError
from ..ports.graph_library import GraphLibrary, normalize_graph_name


class GetGraph:
    def __init__(self, library: GraphLibrary) -> None:
        self._library = library

    def execute(self, name: str) -> SavedGraphDTO:
        clean_name = self._clean(name)
        saved = self._library.get(clean_name)
        if saved is None:
            raise GraphNotFoundError(f"no saved graph named {clean_name!r}")
        return to_saved_graph_dto(saved)

    @staticmethod
    def _clean(name: str) -> str:
        try:
            return normalize_graph_name(name)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
