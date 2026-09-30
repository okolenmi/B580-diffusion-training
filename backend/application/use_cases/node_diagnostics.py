"""NodeDiagnostics -- live per-input diagnostic text for one node class.

``nodes.core.Node.diagnostics()`` is an opt-in, read-only side channel a
caller uses *before* running the graph (per-component dtype of a
resolved checkpoint, rank of a saved LoRA, ...). Same posture as the
legacy endpoint: an unknown class is 404 ``node_class_not_found``; a
node raising because its current params are mid-edit (relative path that
doesn't resolve, missing file) is an ordinary 400
``node_diagnostics_failed``, never a 500.
"""

from __future__ import annotations

from ..errors import ApplicationError, NodeDiagnosticsError
from ..ports.graph_catalog import GraphCatalog


class NodeDiagnostics:
    def __init__(self, catalog: GraphCatalog) -> None:
        self._catalog = catalog

    def execute(self, class_name: str, params: dict) -> dict[str, list[str]]:
        try:
            return self._catalog.diagnostics(class_name, dict(params))
        except ApplicationError:
            raise  # node_class_not_found and friends pass through typed
        except Exception as exc:  # noqa: BLE001 -- the node's own raise is the outcome
            raise NodeDiagnosticsError(f"{type(exc).__name__}: {exc}") from exc
