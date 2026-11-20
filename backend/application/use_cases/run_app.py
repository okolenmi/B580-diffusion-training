"""RunApp -- run a saved app's graph with the values typed into it.

The only way an app starts a graph. It loads the document, applies the submitted values through ``app_binding.apply_values`` (which
refuses anything the app does not expose), and hands the result to the same ``StartGraphExecution`` the editor's Run button uses -- so
validation, single-active, memory admission and every refusal are *the same code path*, not an app-flavoured copy that could drift
from it. The run remembers its origin: which app, which version, with which values.
"""

from __future__ import annotations

from ...domain.graph import GraphDefinition
from ..app_binding import apply_values
from ..dto import GraphExecutionSummaryDTO
from ..errors import AppInvalidError, AppNotFoundError, InvalidQueryError
from ..ports.app_library import AppLibrary, normalize_app_name
from .save_graph import normalize_graph_payload
from .start_graph_execution import StartGraphExecution


class RunApp:
    def __init__(self, *, library: AppLibrary, start: StartGraphExecution) -> None:
        self._library = library
        self._start = start

    def execute(
        self, name: str, values: dict | None = None, *, memory_overrides: dict | None = None, mode: str = "auto"
    ) -> GraphExecutionSummaryDTO:
        try:
            clean = normalize_app_name(name)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        saved = self._library.get(clean)
        if saved is None:
            raise AppNotFoundError(f"no saved app named {clean!r}")
        try:
            graph = GraphDefinition.from_dict(normalize_graph_payload(saved.doc["graph"]))
        except ValueError as exc:
            raise AppInvalidError(f"graph: {exc}", details=[str(exc)]) from exc
        graph, problems = apply_values(graph, saved.doc, values)
        if problems:
            raise AppInvalidError(f"{len(problems)} value(s) refused", details=problems)
        origin = {"app": saved.name, "version": saved.version, "values": values or {}}
        return self._start.execute(graph, memory_overrides=memory_overrides, mode=mode, origin=origin)
