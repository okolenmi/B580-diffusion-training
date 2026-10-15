"""What a family's text conditioning looks like to everything that is not that family.

Pure declarations: no torch import. Tensors appear only as ``Any`` here.
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Conditioning(Protocol):
    """Opaque to the trainer; these operations are all it needs."""

    def to(self, device: Any = None, dtype: Any = None) -> "Conditioning": ...

    def index(self, i: Any) -> "Conditioning": ...

    def blend(self, other: "Conditioning", p: Any, segments: Sequence[str] | None = None) -> "Conditioning": ...

    def segments(self) -> dict[str, Any]: ...

    def nbytes(self) -> int: ...


class ConditioningKit(Protocol):
    def cache_key(self, prompt: str, size: Any = None) -> Hashable: ...

    def encode(self, prompts: Sequence[str], sizes: Sequence[Any] | None = None) -> list[Conditioning]: ...

    def stack(self, items: Sequence[Conditioning]) -> Conditioning: ...

    def blank(self, size: Any = None) -> Conditioning: ...

    def bytes_per_prompt(self) -> int: ...
