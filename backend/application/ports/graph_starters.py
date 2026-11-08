"""GraphStarters -- ready-made graph documents the editor can start from (contract ``starters.md``)."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GraphStarters(ABC):
    @abstractmethod
    def starters(self) -> list[dict]:
        """Each starter: ``name``, ``label``, ``doc`` and its ``options`` (port-like entries: name, label, type, default, choices, picker, doc, advanced)."""
        raise NotImplementedError

    @abstractmethod
    def build(self, name: str, values: dict) -> dict:
        """The format-3 document of one starter for these option values.

        Raises ``StarterNotFoundError`` for an unknown name and ``InvalidQueryError`` for a value the starter refuses."""
        raise NotImplementedError
