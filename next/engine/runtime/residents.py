"""Anything that occupies device memory and can be moved off it: parts, encoders, optimizer state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from torch import Tensor


class DeviceResident(ABC):
    @abstractmethod
    def footprint_bytes(self) -> int:
        """Bytes currently held on the device (0 while offloaded)."""

    @abstractmethod
    def offload(self) -> None:
        """Move everything to the host. Must be reversible with ``reload``."""

    @abstractmethod
    def reload(self, device: str | None = None) -> None:
        """Bring everything back to the device it came from (or ``device``)."""

    @abstractmethod
    def release(self) -> None:
        """Drop everything; the object is unusable afterwards."""


def tensor_bytes(tensors: Iterable[Tensor | None]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)
