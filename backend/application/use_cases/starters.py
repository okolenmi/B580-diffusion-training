"""ListStarters and BuildStarter -- the editor's "new graph from a starter" (contract ``starters.md``)."""

from __future__ import annotations

from ..ports.graph_starters import GraphStarters


class ListStarters:
    def __init__(self, starters: GraphStarters) -> None:
        self._starters = starters

    def execute(self) -> list[dict]:
        return self._starters.starters()


class BuildStarter:
    def __init__(self, starters: GraphStarters) -> None:
        self._starters = starters

    def execute(self, name: str, values: dict) -> dict:
        return self._starters.build(name, values)
