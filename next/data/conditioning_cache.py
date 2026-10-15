"""Text conditioning computed once per distinct prompt and reused across epochs.

The cache holds conditionings on the CPU (the text encoders can then leave the device) and is bounded
by bytes. ``warm`` encodes only what is missing, in batches; ``retain`` drops entries whose prompt no
longer occurs, which is what a ``PromptsChanged`` event asks for.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Iterable, Sequence

from ..families.api.conditioning import Conditioning, ConditioningKit

Size = tuple[int, int]


class ConditioningCache:
    def __init__(self, kit: ConditioningKit, *, max_bytes: int = 4 << 30, batch_size: int = 16):
        self.kit, self.max_bytes, self.batch_size = kit, max_bytes, batch_size
        self._items: "OrderedDict[Any, tuple[str, Conditioning]]" = OrderedDict()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def fits(self, prompts: int) -> bool:
        """Whether ``prompts`` distinct entries stay under the byte budget."""
        return prompts * self.kit.bytes_per_prompt() <= self.max_bytes

    def warm(self, prompts: Iterable[str], size: Size | None = None,
             progress: Callable[[int, int], None] | None = None) -> int:
        """Encode every prompt not yet cached; returns how many were encoded."""
        todo = [p for p in dict.fromkeys(prompts) if self.kit.cache_key(p, size) not in self._items]
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            for prompt, cond in zip(chunk, self.kit.encode(chunk, [size] * len(chunk) if size else None)):
                self._put(self.kit.cache_key(prompt, size), prompt, cond.to("cpu"))
            if progress:
                progress(min(start + self.batch_size, len(todo)), len(todo))
        return len(todo)

    def get(self, prompts: Sequence[str], sizes: Sequence[Size | None] | Size | None = None) -> Conditioning:
        """One stacked conditioning for a batch, encoding any prompt that is missing."""
        per_item = list(sizes) if isinstance(sizes, list) else [sizes] * len(prompts)
        found = []
        for prompt, size in zip(prompts, per_item):
            key = self.kit.cache_key(prompt, size)
            if key not in self._items:
                self.warm([prompt], size)
            self._items.move_to_end(key)
            found.append(self._items[key][1])
        return self.kit.stack(found)

    def retain(self, prompts: set[str]) -> int:
        """Forget entries for prompts outside ``prompts``; returns how many were dropped."""
        stale = [k for k, (text, _) in self._items.items() if text not in prompts]
        for key in stale:
            self._bytes -= self._items.pop(key)[1].nbytes()
        return len(stale)

    def _put(self, key: Any, prompt: str, cond: Conditioning) -> None:
        self._items[key] = (prompt, cond)
        self._bytes += cond.nbytes()
        while self._bytes > self.max_bytes and len(self._items) > 1:
            _, (_, evicted) = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes()
