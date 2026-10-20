"""What each checkpointed block costs: the memory its activations take and the time its forward takes.

In backward a checkpointed block runs its forward again with gradients on, which is exactly the work (and the memory
held for the backward) it would have cost without checkpointing. Timing that recompute and watching allocated memory
around it therefore measures the block for free, in a real step, with no extra profiling pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True)
class BlockCost:
    activation_mb: float  # largest memory held by the block's activations over the steps seen
    recompute_ms: float  # mean time of its forward


class BlockProfiler:
    """Watches the recomputes of ``checkpointer`` while used as a context manager; ``blocks`` are (path, module)."""

    def __init__(self, checkpointer: Any, blocks: Sequence[tuple[str, Any]], memory: Callable[[], float],
                 synchronize: Callable[[], None] = lambda: None):
        self._checkpointer = checkpointer
        self._names = {id(module): name for name, module in blocks}
        self.order = [name for name, _ in blocks]
        self._memory, self._sync = memory, synchronize
        self._stats: dict[str, list[float]] = {}  # name -> [largest mb, summed ms, count]

    def observer(self, run: Callable, args: list) -> Any:
        name = self._names.get(id(getattr(run, "__self__", None)))
        if name is None:
            return run(*args)
        self._sync()
        before, started = self._memory(), time.perf_counter()
        outputs = run(*args)
        self._sync()
        elapsed, after = (time.perf_counter() - started) * 1000, self._memory()
        entry = self._stats.setdefault(name, [0.0, 0.0, 0])
        entry[0], entry[1], entry[2] = max(entry[0], after - before), entry[1] + elapsed, entry[2] + 1
        return outputs

    def __enter__(self) -> "BlockProfiler":
        self._checkpointer.observer = self.observer
        return self

    def __exit__(self, *exc) -> None:
        self._checkpointer.observer = None

    def reset(self) -> None:
        self._stats.clear()

    def costs(self) -> dict[str, BlockCost]:
        return {name: BlockCost(mb, ms / count) for name, (mb, ms, count) in self._stats.items()}
