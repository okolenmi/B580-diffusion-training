"""HostMemory port -- what the machine's RAM looks like right now.

Admission reads it to refuse a run whose stated host-memory ceiling the machine cannot give. ``None`` means the platform
cannot say (never a zero): the check is then skipped, not failed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HostRam:
    total_mb: float
    #: What a new process can take without pushing others to swap (the kernel's own estimate, not "free").
    available_mb: float


class HostMemory(ABC):
    @abstractmethod
    def snapshot(self) -> HostRam | None:
        """The machine's RAM now, or None when it cannot be read here."""
        raise NotImplementedError
