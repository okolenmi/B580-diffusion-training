"""ValidateGraph -- dry-run check of a graph, nothing executed.

Returns *every* finding (errors and warnings) so the editor can mark
all problems at once; ``ok`` means no error-severity issue exists.
``StartGraphExecution`` runs the exact same port call and rejects on the
errors -- one validation path, not two.
"""

from __future__ import annotations

from ..dto import GraphValidationResult
from ..ports.graph_runtime import GraphRuntime, IssueSeverity
from ...domain.graph import GraphDefinition


class ValidateGraph:
    def __init__(self, runtime: GraphRuntime) -> None:
        self._runtime = runtime

    def execute(self, graph: GraphDefinition) -> GraphValidationResult:
        issues = self._runtime.validate(graph)
        ok = not any(IssueSeverity(issue.severity).blocks for issue in issues)
        return GraphValidationResult(ok=ok, issues=issues)
