"""Unit tests -- the PID-reuse guard, and the pre-exec window inside it.

`cmdline_mentions` is the only thing deciding whether a signal goes to a
pid we started or to a stranger holding the same number (docs 07 F-12),
so its three-valued answer is pinned directly here rather than only
through the gateways.

The interesting case is the one no test covered: between `fork` and
`execve` a child's `/proc/<pid>/cmdline` still reads back as its
*parent's* command line. Answering that "does not mention our marker"
reports a live trainer as finished.

Run directly: python backend/tests/test_process_identity.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.infrastructure.process_identity import (
    _parent_pid,
    _read_cmdline,
    cmdline_mentions,
)
from backend.tests.support import check, finish

#: Not a substring of anything a test process is invoked with.
ABSENT = "zzz-marker-that-is-in-no-cmdline-zzz"
#: In this test file's own command line, and so in any fork of it that has
#: not exec'd yet. That is the point: the pre-exec child *does* mention it.
PRESENT = "test_process_identity.py"


def _forked_child() -> tuple[int, int]:
    """A forked child that has NOT exec'd, plus the fd that releases it.

    `os.fork` without `exec` holds the pre-exec state open indefinitely,
    which is what makes this testable at all: `Popen` closes the window in
    microseconds, so a test built on it is a coin flip rather than a
    regression test.

    Returns the child pid and the **write** end of a pipe the child is
    blocked reading. The caller must pass it to `_release`.
    """
    hold_fd, release_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover -- the child never returns
        os.close(release_fd)
        try:
            os.read(hold_fd, 1)  # block until the parent lets us go
        finally:
            os.close(hold_fd)
        os._exit(0)  # not exit(): this process shares the parent's buffers
    os.close(hold_fd)
    return pid, release_fd


def _release(pid: int, release_fd: int) -> None:
    os.write(release_fd, b"x")
    os.close(release_fd)
    os.waitpid(pid, 0)


def test_pre_exec_child_is_cannot_tell() -> None:
    print("\n== a child that has not exec'd yet: cannot tell, not no ==")
    pid, release_fd = _forked_child()
    try:
        inherited = _read_cmdline(pid)
        check(inherited is not None and inherited == _read_cmdline(os.getpid()),
              f"the forked child's cmdline is still its parent's "
              f"(got {inherited!r})")

        verdict = cmdline_mentions(pid, ABSENT)
        check(verdict is None,
              f"a marker absent from that cmdline means 'not yet know', "
              f"not 'not ours' (got {verdict!r}) -- the old answer was "
              f"False, which reports a live trainer as finished")

        check(cmdline_mentions(pid, PRESENT) is True,
              "a marker it does contain is still a match -- the parent's "
              "command line genuinely says it")
    finally:
        _release(pid, release_fd)


def test_genuinely_foreign_process_is_still_false() -> None:
    print("\n== a process that is not ours: still a firm no ==")
    sleeper = subprocess.Popen(["/bin/sleep", "30"])
    try:
        _wait_for_cmdline(sleeper.pid)
        check(_read_cmdline(sleeper.pid) != _read_cmdline(os.getpid()),
              "it has exec'd, so its cmdline is its own")
        check(cmdline_mentions(sleeper.pid, ABSENT) is False,
              "a real mismatch is still False -- the guard that refuses to "
              "signal a stranger must not be softened")
        check(cmdline_mentions(sleeper.pid, "sleep") is True,
              "and a real match is still True")
    finally:
        sleeper.kill()
        sleeper.wait()


def test_vanished_and_zombie() -> None:
    print("\n== gone, and not gone ==")
    child = subprocess.Popen(["/bin/true"])
    child.wait()  # reaped: the pid is free, and /proc has no entry
    check(cmdline_mentions(child.pid, ABSENT) is None,
          "a pid with no /proc entry cannot be judged")

    # Not reaped: a zombie's cmdline reads as empty. Empty is not
    # "inherited from the parent", so this stays a firm no -- the
    # pre-exec exception must not leak into processes that are on their
    # way out.
    zombie = subprocess.Popen(["/bin/true"])
    zombie_pid = zombie.pid
    # Waited for, not slept on. A fixed sleep was a latent flake here:
    # it assumed /bin/true exits within 50ms, which holds on an idle
    # machine and stops holding the moment the suite has real work running
    # alongside it -- and it failed the *whole* file, on an assertion
    # about a process that had simply not exited yet.
    check(_wait_until(lambda: _read_cmdline(zombie_pid) == ""),
          f"the unreaped child becomes a zombie, whose cmdline reads empty "
          f"(got {_read_cmdline(zombie_pid)!r})")
    check(_read_cmdline(zombie_pid) == "",
          f"the unreaped child has an empty cmdline (got "
          f"{_read_cmdline(zombie_pid)!r})")
    check(cmdline_mentions(zombie_pid, ABSENT) is False,
          "a zombie is a firm no, as documented -- it is not pre-exec")
    zombie.wait()


def test_parent_pid_parsing() -> None:
    print("\n== ppid comes out of /proc/<pid>/stat, whose name may contain junk ==")
    check(_parent_pid(os.getpid()) == os.getppid(),
          f"a plain process parses (got {_parent_pid(os.getpid())})")

    pid, release_fd = _forked_child()
    try:
        check(_parent_pid(pid) == os.getpid(),
              f"the forked child's parent is us (got {_parent_pid(pid)})")
    finally:
        _release(pid, release_fd)

    check(_parent_pid(999_999_999) is None,
          "a pid with no /proc entry has no readable parent")


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _wait_for_cmdline(pid: int, timeout: float = 5.0) -> str | None:
    """Poll until the pid's cmdline is its own, not the parent's.

    The window closes on its own; waiting for it is what makes the
    assertions above deterministic rather than lucky.
    """
    own = _read_cmdline(os.getpid())
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = _read_cmdline(pid)
        if current is not None and current != own:
            return current
        time.sleep(0.005)
    return _read_cmdline(pid)


def main() -> None:
    test_pre_exec_child_is_cannot_tell()
    test_genuinely_foreign_process_is_still_false()
    test_vanished_and_zombie()
    test_parent_pid_parsing()
    finish()


if __name__ == "__main__":
    main()