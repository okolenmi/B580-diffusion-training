"""Anything that occupies device memory and can be moved off it: parts, encoders, optimizer state."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Sequence

from torch import Tensor, nn


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


class ModulesResident(DeviceResident):
    """Adapts anything that owns modules and a ``to(device)`` method (text encoders, VAE codecs)."""

    def __init__(self, modules: Sequence[nn.Module], move: Callable[[str], None], home: str):
        self.modules, self.move, self.home = tuple(modules), move, home

    def footprint_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for m in self.modules for p in m.parameters() if p.device.type != "cpu")

    def offload(self) -> None:
        self.move("cpu")

    def reload(self, device: str | None = None) -> None:
        self.move(device or self.home)

    def release(self) -> None:
        self.modules = ()
