"""Brings the server's console window back when something goes wrong while it is minimised.

The server runs in a console the user usually parks out of the way; an error that only lands there goes unseen. On Windows
the window is restored without taking focus. Elsewhere nothing happens: a terminal is not something a process may raise.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Protocol

# ShowWindow's "restore, but do not activate": the user's typing is not interrupted.
_SW_SHOWNOACTIVATE = 4
# GetAncestor's root owner: under Windows Terminal the console handle is a hidden pseudo window owned by the terminal.
_GA_ROOTOWNER = 3


class WindowApi(Protocol):
    """The four window calls the reveal needs; the Windows one is `Win32Window`, tests pass a fake."""

    def console_window(self) -> int: ...
    def top_owner(self, handle: int) -> int: ...
    def is_minimised(self, handle: int) -> bool: ...
    def restore(self, handle: int) -> None: ...


class Win32Window:
    def __init__(self) -> None:
        import ctypes  # noqa: PLC0415 -- only meaningful on Windows

        self._kernel = ctypes.windll.kernel32
        self._user = ctypes.windll.user32
        # Handles are pointer sized; the default int return type would truncate them.
        self._kernel.GetConsoleWindow.restype = ctypes.c_void_p
        self._user.GetAncestor.restype = ctypes.c_void_p
        self._user.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._user.IsIconic.argtypes = [ctypes.c_void_p]
        self._user.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def console_window(self) -> int:
        return self._kernel.GetConsoleWindow() or 0

    def top_owner(self, handle: int) -> int:
        return self._user.GetAncestor(handle, _GA_ROOTOWNER) or handle

    def is_minimised(self, handle: int) -> bool:
        return bool(self._user.IsIconic(handle))

    def restore(self, handle: int) -> None:
        self._user.ShowWindow(handle, _SW_SHOWNOACTIVATE)


class ConsoleReveal:
    """Restores the console once per `quiet_s`, so a burst of errors does not undo a user who minimised it again."""

    def __init__(self, api: WindowApi | None, quiet_s: float = 30.0, clock=time.monotonic) -> None:
        self._api = api
        self._quiet_s = quiet_s
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    @classmethod
    def for_this_process(cls) -> ConsoleReveal:
        return cls(Win32Window() if sys.platform == "win32" else None)

    def reveal(self) -> bool:
        """True when a minimised window was restored."""
        if self._api is None:
            return False
        with self._lock:
            now = self._clock()
            if self._last is not None and now - self._last < self._quiet_s:
                return False
            try:
                handle = self._api.console_window()
                if not handle:
                    return False
                handle = self._api.top_owner(handle)
                if not self._api.is_minimised(handle):
                    return False
                self._api.restore(handle)
            except Exception:  # noqa: BLE001 -- a failing window call must never break the log call that triggered it
                return False
            self._last = now
            return True


class RevealOnError(logging.Handler):
    """A log handler that shows the console for records at ERROR or above."""

    def __init__(self, reveal: ConsoleReveal) -> None:
        super().__init__(level=logging.ERROR)
        self._reveal = reveal

    def emit(self, record: logging.LogRecord) -> None:
        self._reveal.reveal()


def install(reveal: ConsoleReveal, loggers: tuple[str, ...] = ("", "uvicorn.error")) -> None:
    """Attach to the given loggers and to uncaught exceptions of threads. uvicorn's logger does not propagate to the root one,
    and its own configuration drops handlers set before it ran, so call this after uvicorn's `Config` exists."""
    handler = RevealOnError(reveal)
    for name in loggers:
        logging.getLogger(name).addHandler(handler)
    previous = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        reveal.reveal()
        previous(args)

    threading.excepthook = hook
