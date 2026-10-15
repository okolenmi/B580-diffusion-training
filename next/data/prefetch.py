"""Run a batch-producing iterator on a worker thread, a few batches ahead of the consumer."""

from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable, Iterator

_END = object()


class Prefetch:
    """One pass over ``make()``'s items. A pass owns its worker: closing it stops the thread, and nothing
    of a later pass can be read ahead by this one."""

    def __init__(self, make: Callable[[], Iterable], depth: int = 2):
        if depth < 1:
            raise ValueError("prefetch depth must be at least 1")
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(make,), daemon=True, name="prefetch")
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, make: Callable[[], Iterable]) -> None:
        try:
            for item in make():
                if not self._put(item):
                    return
            self._put(_END)
        except BaseException as exc:  # handed to the consumer, which re-raises it
            self._put(exc)

    def __iter__(self) -> Iterator:
        while True:
            item = self._queue.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
