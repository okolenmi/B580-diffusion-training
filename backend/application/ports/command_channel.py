"""CommandChannel port -- delivering a page's action to a run that may live in another process."""

from __future__ import annotations

from abc import ABC, abstractmethod


class CommandChannel(ABC):
    @abstractmethod
    def submit(self, execution_id: int, node: str, action: str) -> str:
        """Queue the command for the run and return its id (the answer frames on ``node:<uid>`` carry it)."""
