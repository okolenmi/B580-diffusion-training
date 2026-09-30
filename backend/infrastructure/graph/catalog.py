"""DiscoveredGraphCatalog -- the GraphCatalog port over auto-discovery.

The palette's read path: snapshot (every discovered node, grouped
client-side from the ``domain`` field) and per-class diagnostics.
Discovery failures come back as ``load_errors`` in the snapshot, never
as an exception -- one broken module must not take the palette down.
"""

from __future__ import annotations

from ...application.errors import NodeClassNotFoundError
from ...application.ports.graph_catalog import (
    CatalogSnapshot,
    GraphCatalog,
    NodeInfo,
)
from .discovery import NodeRegistry
from .introspect import introspect_node_class


def domain_of(cls: type) -> str:
    """nodes.dataset.managed -> 'dataset'. Derived from the module path,
    not hand-labeled, so a node can't land in the wrong palette group
    because someone forgot to update a second list."""
    parts = cls.__module__.split(".")
    return parts[1] if len(parts) > 1 else "other"


class DiscoveredGraphCatalog(GraphCatalog):
    def __init__(self, registry: NodeRegistry) -> None:
        self._registry = registry

    def snapshot(self, *, refresh: bool = False) -> CatalogSnapshot:
        classes, errors = self._registry.load(refresh=refresh)
        nodes: list[NodeInfo] = [
            introspect_node_class(cls, domain=domain_of(cls))
            for cls in classes.values()
        ]
        return CatalogSnapshot(nodes=tuple(nodes), load_errors=errors)

    def diagnostics(self, class_name: str, params: dict) -> dict[str, list[str]]:
        classes, _ = self._registry.load()
        cls = classes.get(class_name)
        if cls is None:
            raise NodeClassNotFoundError(f"unknown node class {class_name!r}")
        # Fresh instance per call (Node.__init__ is cheap; nothing
        # meaningful is cached on self) -- same posture as legacy.
        return cls().diagnostics(dict(params))
