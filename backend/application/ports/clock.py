"""Clock port -- time is an injected dependency, never a direct call."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime


class Clock(ABC):
    """Supplies the current instant (timezone-aware, UTC)."""

    @abstractmethod
    def now(self) -> datetime:
        """Return the current time."""
        raise NotImplementedError
