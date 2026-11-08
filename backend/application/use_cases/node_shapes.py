"""GetNodeCatalog, ListModelFiles and ResolveNode -- the editor's read side over the node tree (contracts ``catalog.md`` and ``resolve.md``)."""

from __future__ import annotations

from ..ports.node_catalog import NodeCatalog


class GetNodeCatalog:
    def __init__(self, catalog: NodeCatalog) -> None:
        self._catalog = catalog

    def execute(self, *, refresh: bool = False) -> dict:
        return self._catalog.catalog(refresh=refresh)


class ListModelFiles:
    def __init__(self, catalog: NodeCatalog) -> None:
        self._catalog = catalog

    def execute(self, scope: str) -> list[str]:
        return self._catalog.model_files(scope)


class ResolveNode:
    def __init__(self, catalog: NodeCatalog) -> None:
        self._catalog = catalog

    def execute(self, class_name: str, params: dict, settings: dict | None = None) -> dict:
        return self._catalog.resolve(class_name, dict(params), dict(settings or {}))
