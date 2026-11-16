"""Moving the running server into the environment the user chose.

The server first runs in a small bootstrap environment that has only its own four packages. The node library imports torch, so a
server left there sees almost no nodes. Once setup has chosen an environment, the server restarts into it; the same rule applies at
every later start through the launcher.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ServerRestarter(ABC):
    @abstractmethod
    def running_in(self, interpreter: str) -> bool:
        """Whether this server already runs under ``interpreter``."""

    @abstractmethod
    def can_serve(self, interpreter: str) -> bool:
        """Whether ``interpreter`` has the packages the server itself imports. A restart into one that does not would end the session."""

    @abstractmethod
    def restart_into(self, interpreter: str) -> None:
        """Replace this process with the server running under ``interpreter``, shortly after the current response is sent."""
