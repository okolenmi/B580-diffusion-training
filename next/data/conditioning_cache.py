"""Text conditioning computed once per distinct prompt and reused across epochs.

The cache holds conditionings on the CPU (the text encoders can then leave the device) and is bounded
by bytes. ``warm`` encodes only what is missing, in batches; ``retain`` drops entries whose prompt no
longer occurs, which is what a ``PromptsChanged`` event asks for.

With a ``store``, encodings of STABLE prompts are read from disk when present and written after being computed;
FLEETING prompts (see ``prompts.Lifetime``) live in memory only, for as long as their epoch needs them.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from collections.abc import Callable, Iterable, Sequence

from ..families.api.conditioning import Conditioning, ConditioningKit
from .conditioning_store import ConditioningStore
from .prompts import Lifetime, lifetime_of

Size = tuple[int, int]


class ConditioningCache:
    """Caches the kit's size-independent text encodings; the per-image size is joined in when a batch asks."""

    def __init__(self, kit: ConditioningKit, *, max_bytes: int = 4 << 30, batch_size: int = 16,
                 store: ConditioningStore | None = None):
        self.kit, self.max_bytes, self.batch_size, self.store = kit, max_bytes, batch_size, store
        self.loaded = 0  # entries read from the store instead of encoded
        self._items: OrderedDict[Any, tuple[str, Any]] = OrderedDict()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def fits(self, prompts: int) -> bool:
        """Whether ``prompts`` distinct entries stay under the byte budget."""
        return prompts * self.kit.bytes_per_prompt() <= self.max_bytes

    def warm(self, prompts: Iterable[str], progress: Callable[[int, int], None] | None = None) -> int:
        """Make every prompt available; returns how many had to be encoded (the rest came from memory or the store)."""
        todo = [p for p in dict.fromkeys(prompts) if self.kit.cache_key(p) not in self._items]
        if self.store is not None:
            missing = []
            for prompt in todo:
                text = self.store.get(prompt) if lifetime_of(prompt) is Lifetime.STABLE else None
                if text is None:
                    missing.append(prompt)
                else:
                    self._put(self.kit.cache_key(prompt), prompt, text)
                    self.loaded += 1
            todo = missing
        for start in range(0, len(todo), self.batch_size):
            chunk = todo[start:start + self.batch_size]
            encoded = [t.to("cpu") for t in self.kit.encode_text(chunk)]
            for prompt, text in zip(chunk, encoded):
                self._put(self.kit.cache_key(prompt), prompt, text)
            if self.store is not None:
                self.store.put_many([(p, t) for p, t in zip(chunk, encoded) if lifetime_of(p) is Lifetime.STABLE])
            if progress:
                progress(min(start + self.batch_size, len(todo)), len(todo))
        return len(todo)

    def get(self, prompts: Sequence[str], sizes: Sequence[Size | None] | Size | None = None) -> Conditioning:
        """One stacked conditioning for a batch (``sizes``: one per prompt, or one for all), encoding what is missing."""
        per_item = list(sizes) if isinstance(sizes, list) else [sizes] * len(prompts)
        found = []
        for prompt, size in zip(prompts, per_item):
            key = self.kit.cache_key(prompt)
            if key not in self._items:
                self.warm([prompt])
            self._items.move_to_end(key)
            found.append(self.kit.assemble(self._items[key][1], size))
        return self.kit.stack(found)

    def retain(self, prompts: set[str]) -> int:
        """Forget entries for prompts outside ``prompts``; returns how many were dropped."""
        stale = [k for k, (text, _) in self._items.items() if text not in prompts]
        for key in stale:
            self._bytes -= self._items.pop(key)[1].nbytes()
        return len(stale)

    def _put(self, key: Any, prompt: str, text: Any) -> None:
        self._items[key] = (prompt, text)
        self._bytes += text.nbytes()
        while self._bytes > self.max_bytes and len(self._items) > 1:
            _, (_, evicted) = self._items.popitem(last=False)
            self._bytes -= evicted.nbytes()
