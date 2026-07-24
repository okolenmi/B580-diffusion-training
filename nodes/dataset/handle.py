"""Runtime contract for anything that yields training batches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class TrainingBatchSource(ABC):

    @abstractmethod
    def __iter__(self) -> Iterator[dict]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def invalidate(self) -> None:
        """Drop any cached samples; next iteration reloads from source."""
