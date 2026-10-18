"""What a family's text conditioning looks like to everything that is not that family.

Pure declarations: no torch import. Tensors appear only as ``Any`` here.
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol, Sequence, runtime_checkable


@runtime_checkable
class Conditioning(Protocol):
    """Opaque to the trainer; these operations are all it needs."""

    def to(self, device: Any = None, dtype: Any = None) -> "Conditioning": ...

    def tensors(self) -> list[Any]:
        """Every tensor it holds, in a fixed order."""

    def map(self, fn: Any) -> "Conditioning":
        """The same conditioning with ``fn`` applied to each tensor, in the order ``tensors`` lists them."""

    def index(self, i: Any) -> "Conditioning": ...

    def blend(self, other: "Conditioning", p: Any, segments: Sequence[str] | None = None) -> "Conditioning": ...

    def segments(self) -> dict[str, Any]: ...

    def nbytes(self) -> int: ...


class ConditioningKit(Protocol):
    def cache_key(self, prompt: str) -> Hashable: ...

    def encode_text(self, prompts: Sequence[str]) -> list[Any]:
        """The size-independent encodings worth caching; each has ``to(device, dtype)`` and ``nbytes()``."""

    def dump_text(self, text: Any) -> dict[str, Any]:
        """The tensors of a cached text encoding, by name, for a disk store."""

    def load_text(self, tensors: dict[str, Any]) -> Any:
        """The text encoding ``dump_text`` made these tensors from."""

    def assemble(self, text: Any, size: Any = None) -> Conditioning:
        """One conditioning from a cached text encoding and the image size it is for."""

    def encode(self, prompts: Sequence[str], sizes: Sequence[Any] | None = None) -> list[Conditioning]: ...

    def stack(self, items: Sequence[Conditioning]) -> Conditioning: ...

    def blank(self, size: Any = None) -> Conditioning: ...

    def bytes_per_prompt(self) -> int: ...
