"""NodeCatalog port -- what the editor learns about node classes (rework/contracts/catalog.md, resolve.md).

The answers are the contract's JSON documents, passed through as plain dictionaries: the catalog is
defined once, in ``next/graph/catalog.py``, and the server adds nothing to it. Nothing here runs a node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class NodeCatalog(ABC):
    @abstractmethod
    def catalog(self, *, refresh: bool = False) -> dict:
        """``GET /catalog``: nodes, settings, handle types, families, load errors. ``refresh`` scans the node packages again."""
        raise NotImplementedError

    @abstractmethod
    def resolve(self, class_name: str, params: dict, settings: dict) -> dict:
        """One node's shape for these params (contract ``resolve.md``).

        Raises ``NodeClassNotFoundError`` for an unknown class; bad params are an ``error`` state in the answer, never an exception.
        """
        raise NotImplementedError
