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

    def prompt_universe(self) -> list[str]:
        """Every prompt this source can serve that one pass over it might not
        show -- the text-encoder prewarm warms these too. Concrete, and empty,
        because most sources serve exactly the prompts one pass shows; a
        source that draws from a bounded pool of prompts (a caption pool)
        overrides it."""
        return []
