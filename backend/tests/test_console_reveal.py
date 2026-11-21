"""Console reveal: the server's console window comes back when an error is logged while it is minimised.

  A  the rule: only a minimised window is restored, the owner window is the one asked, calls are throttled, failures are silent
  B  the log handler: ERROR reaches it, lower levels do not; no window api (not Windows) does nothing

Run directly: python backend/tests/test_console_reveal.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.console_reveal import ConsoleReveal, RevealOnError

PASSED = FAILED = 0


def check(name: str, ok: bool) -> None:
    global PASSED, FAILED
    if ok:
        PASSED += 1
    else:
        FAILED += 1
        print(f"FAIL {name}")


class FakeWindow:
    def __init__(self, minimised: bool = True, handle: int = 7, owner: int = 9, boom: bool = False) -> None:
        self.minimised, self.handle, self.owner, self.boom = minimised, handle, owner, boom
        self.asked: list[int] = []
        self.restored: list[int] = []

    def console_window(self) -> int:
        if self.boom:
            raise OSError("no window")
        return self.handle

    def top_owner(self, handle: int) -> int:
        return self.owner

    def is_minimised(self, handle: int) -> bool:
        self.asked.append(handle)
        return self.minimised

    def restore(self, handle: int) -> None:
        self.restored.append(handle)


def rule() -> None:
    api = FakeWindow()
    now = [0.0]
    reveal = ConsoleReveal(api, quiet_s=30, clock=lambda: now[0])
    check("minimised window restored", reveal.reveal() is True)
    check("the owner window is the one asked and restored", api.asked == [9] and api.restored == [9])
    now[0] = 10
    check("throttled inside the quiet time", reveal.reveal() is False and len(api.restored) == 1)
    now[0] = 31
    check("restored again after it", reveal.reveal() is True and len(api.restored) == 2)

    open_window = FakeWindow(minimised=False)
    check("a visible window is left alone", ConsoleReveal(open_window).reveal() is False and not open_window.restored)
    check("no console handle does nothing", ConsoleReveal(FakeWindow(handle=0)).reveal() is False)
    check("a failing call is swallowed", ConsoleReveal(FakeWindow(boom=True)).reveal() is False)


def handler() -> None:
    api = FakeWindow()
    log = logging.getLogger("console_reveal_test")
    log.propagate = False
    log.setLevel(logging.DEBUG)
    log.addHandler(RevealOnError(ConsoleReveal(api)))
    log.warning("not an error")
    check("a warning does not reveal", not api.restored)
    log.error("an error")
    check("an error reveals", api.restored == [9])
    check("no window api is a no-op", ConsoleReveal(None).reveal() is False)


if __name__ == "__main__":
    rule()
    handler()
    print(f"{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
