"""Process identity -- one implementation of the PID-reuse guard.

Three subprocess gateways have asked the same two questions, and both
answers live here because both are answered by reading ``/proc``:

* *is this pid still the process we started, or has the number been
  recycled?* (``cmdline_mentions``) Every gateway asks before it signals
  anything, because signals do not come with a name, so the guard is the
  only thing standing between a stale row and an unrelated process on the
  same machine (docs 07 F-12).
* *which pids are running our child, and for which run?*
  (``find_by_argv``) The graph gateway asks this after a restart, to
  adopt a run whose server died but whose child did not.

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
import os
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


def argv_of(pid: int) -> list[str] | None:
    """``/proc/<pid>/cmdline`` split into argv, or ``None`` if unreadable.

    Split on NUL because that is what the kernel writes: argv entries are
    NUL-terminated, so one entry per element with no escaping to reason
    about. An empty argv is a real answer, not a failure -- that is what a
    zombie reads as, and "the process is gone as far as we are concerned"
    is exactly the right reading of one.
    """
    raw = _read_cmdline(pid)
    if raw is None:
        return None
    return [part for part in raw.split("\x00") if part]


def find_by_argv(marker: str, *pairs: str) -> list[int]:
    """Live pids whose argv contains ``marker`` and every ``(flag, value)``.

    Exact token matching rather than a substring test, which is the whole
    reason this takes argv and not a joined string: ``--execution 1`` must
    not match the child running ``--execution 15``. Joining the cmdline
    and searching for ``"1"`` would adopt somebody else's run on the first
    id past nine, and an adopted run gets its history replayed into a row
    that is not its own.

    Pids come back in ascending order, so a caller with more than one
    match gets a deterministic answer rather than whichever the
    directory happened to list first.

    A pid whose argv cannot be read is skipped, not guessed at: it cannot
    be confirmed, and the alternative -- treating an unreadable process as
    a candidate -- would adopt things on the strength of a read failure.
    """
    if len(pairs) % 2:
        raise ValueError(
            f"pairs must be (flag, value) tuples, got {pairs!r}"
        )
    found: list[int] = []
    for entry in _proc_pids():
        argv = argv_of(entry)
        if argv is None or marker not in argv:
            continue
        if all(_has_pair(argv, flag, value) for flag, value in _pairs(pairs)):
            found.append(entry)
    return found


def _pairs(flat: tuple[str, ...]):
    for index in range(0, len(flat), 2):
        yield flat[index], flat[index + 1]


def _has_pair(argv: list[str], flag: str, value: str) -> bool:
    """Is ``flag value`` an adjacent pair somewhere in ``argv``?"""
    return any(
        argv[index] == flag and argv[index + 1] == value
        for index in range(len(argv) - 1)
    )


def _proc_pids() -> list[int]:
    try:
        entries = os.listdir("/proc")
    except OSError as exc:
        logger.warning("cannot list /proc (%s); no process discovery", exc)
        return []
    pids: list[int] = []
    for entry in entries:
        if entry.isdigit():
            pids.append(int(entry))
    pids.sort()
    return pids


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