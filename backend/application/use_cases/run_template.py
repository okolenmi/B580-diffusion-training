"""RunTemplate -- run a saved page's graph with the values typed into it.

The only way a template starts a graph. It loads the document, applies the
submitted values through ``template_binding.apply_values`` (which refuses
anything the page does not expose), and hands the result to the same
``StartGraphExecution`` the editor's Run button uses -- so validation,
single-active, memory admission and every refusal are *the same code path*,
not a template-flavoured copy that could drift from it.
"""

from __future__ import annotations

from ..dto import GraphExecutionSummaryDTO
from ..errors import InvalidQueryError, TemplateInvalidError, TemplateNotFoundError
from ..ports.template_library import TemplateLibrary, normalize_template_name
from ..template_binding import apply_values
from ...domain.graph import GraphDefinition
from .save_graph import normalize_graph_payload
from .start_graph_execution import StartGraphExecution


class RunTemplate:
    def __init__(self, *, library: TemplateLibrary, start: StartGraphExecution) -> None:
        self._library = library
        self._start = start

    def execute(
        self, name: str, values: dict | None = None, *, memory_overrides: dict | None = None
    ) -> GraphExecutionSummaryDTO:
        try:
            clean = normalize_template_name(name)
        except ValueError as exc:
            raise InvalidQueryError(str(exc)) from exc
        saved = self._library.get(clean)
        if saved is None:
            raise TemplateNotFoundError(f"no saved template named {clean!r}")
        if saved.kind != "template":
            raise TemplateInvalidError(
                f"{clean!r} is a {saved.kind} window; only a template can be run",
                details=[f"kind: {saved.kind}"],
            )
        graph_doc = saved.doc.get("graph")
        if graph_doc is None:
            raise TemplateInvalidError(f"{clean!r} has no graph to run", details=["graph: missing"])
        try:
            graph = GraphDefinition.from_dict(normalize_graph_payload(graph_doc))
        except ValueError as exc:
            raise TemplateInvalidError(f"graph: {exc}", details=[str(exc)]) from exc
        graph, problems = apply_values(graph, saved.doc, values)
        if problems:
            raise TemplateInvalidError(
                f"{len(problems)} value(s) refused", details=problems
            )
        return self._start.execute(graph, memory_overrides=memory_overrides)
