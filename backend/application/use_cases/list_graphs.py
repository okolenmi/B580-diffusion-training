"""ListGraphs -- library summaries (no payloads), most recently
updated first -- what a picker renders before anyone opens one."""

from __future__ import annotations

from ..dto import SavedGraphListResult, to_saved_graph_summary
from ..ports.graph_library import GraphLibrary


class ListGraphs:
    def __init__(self, library: GraphLibrary) -> None:
        self._library = library

    def execute(self) -> SavedGraphListResult:
        summaries = tuple(
            to_saved_graph_summary(saved) for saved in self._library.list_graphs()
        )
        return SavedGraphListResult(graphs=summaries, count=len(summaries))
