"""Process identity -- one implementation of the PID-reuse guard.

Both subprocess gateways (training and dataset tasks) answer the same
question before they signal anything: *is this pid still the process we
started, or has the number been recycled?* Signals do not come with a
name, so the guard is the only thing standing between a stale row and an
unrelated process on the same machine (docs 07 F-12).

The contract is three-valued on purpose:

* ``True``  -- ``/proc`` answered and the cmdline mentions our marker.
* ``False`` -- ``/proc`` answered and it does not (this includes zombies,
  whose cmdline is empty: the process is gone as far as we are concerned).
* ``None``  -- ``/proc`` could not answer, **or the cmdline read is not yet
  this process's own** (see ``cmdline_mentions``). The callers decide what
  to do with that; the legacy server failed open, and so does the training
  gateway, because refusing to stop a run whose marker cannot be read
  would strand it forever.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cmdline_mentions(pid: int, marker: str) -> bool | None:
    """Does ``/proc/<pid>/cmdline`` mention ``marker``? See module docstring.

    One subtlety the three values exist to cover, and the reason a
    negative answer is not simply "the string is not there":

    Between ``fork`` and ``execve`` the child has the *same* memory as
    its parent, so ``/proc/<child>/cmdline`` reads back as a copy of the
    **parent's** command line. ``Popen`` can return during that window, so
    a process that is alive, is ours, and is three microseconds from
    becoming correct reads as a command line that mentions nothing we
    recognise.

    Returned as ``False``, that is indistinguishable from a recycled pid
    -- and the caller reports a live trainer as finished, which is the one
    answer this whole module exists to avoid being wrong about.

    So a negative match that is still the parent's own command line is
    reported as ``None``: nothing is known about *this* process yet.
    Comparing against the parent's cmdline is exact rather than a timing
    guess, because that copy is the defining property of the pre-exec
    state. ``None`` also errs toward alive, which is the documented
    fail-safe for an unreadable marker.
    """
    cmdline = _read_cmdline(pid)
    if cmdline is None:
        return None
    if marker in cmdline:
        return True
    if _cmdline_is_inherited(pid, cmdline):
        logger.debug(
            "pid %s has not exec'd yet (cmdline still its parent's): "
            "cannot judge identity", pid,
        )
        return None
    return False


def _read_cmdline(pid: int) -> str | None:
    """The raw cmdline of ``pid``, or ``None`` if it cannot be read."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return None  # no /proc entry: cannot tell (or the pid is gone)
    try:
        return cmdline_path.read_bytes().decode(errors="replace")
    except OSError as exc:
        logger.debug("cannot read cmdline of pid %s: %s", pid, exc)
        return None


def _cmdline_is_inherited(pid: int, cmdline: str) -> bool:
    """Is ``cmdline`` still the parent of ``pid``, i.e. is it pre-exec?

    Compares against the *parent's* cmdline as read now, against the
    ``cmdline`` captured earlier. If the parent is still what it was, the
    child has not exec'd since the capture. A process that exec'd in the
    gap between the two reads is therefore still reported as ``None``,
    which is safe; the only way to be wrong here is to answer ``False``
    for a process that has since become ours, and that cannot happen.
    """
    parent = _parent_pid(pid)
    if parent is None or parent <= 0:
        return False  # no parent to compare against: not the pre-exec state
    return _read_cmdline(parent) == cmdline


def _parent_pid(pid: int) -> int | None:
    """``ppid`` from ``/proc/<pid>/stat``, or ``None`` if unreadable.

    Field 2 is the executable name in parentheses and may itself contain
    spaces and parentheses, so the fields are taken from after the *last*
    ``)`` rather than by splitting the whole line. What follows the last
    ``)`` starts at field 3 (state), making ``ppid`` the second token.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(errors="replace")
    except OSError as exc:
        logger.debug("cannot read stat of pid %s: %s", pid, exc)
        return None
    close = stat.rfind(")")
    if close == -1:
        return None
    fields = stat[close + 1:].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None