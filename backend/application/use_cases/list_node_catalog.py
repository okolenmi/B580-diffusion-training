"""ListNodeCatalog -- the editor palette, introspected on demand.

Thin by design: the interesting logic (discovery, reflection) lives
behind the ``GraphCatalog`` port so it can be swapped or faked without
touching application code. ``refresh=True`` re-scans ``nodes/`` (picks
up newly added node files without a restart); otherwise the registry's
cached scan is served.
"""

from __future__ import annotations

from ..ports.graph_catalog import CatalogSnapshot, GraphCatalog


class ListNodeCatalog:
    def __init__(self, catalog: GraphCatalog) -> None:
        self._catalog = catalog

    def execute(self, *, refresh: bool = False) -> CatalogSnapshot:
        return self._catalog.snapshot(refresh=refresh)
