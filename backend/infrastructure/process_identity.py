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
* ``None``  -- ``/proc`` is unavailable, so nothing can be proven. The
  callers decide what to do with that; the legacy server failed open,
  and so does the training gateway, because refusing to stop a run whose
  marker cannot be read would strand it forever.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def cmdline_mentions(pid: int, marker: str) -> bool | None:
    """Does ``/proc/<pid>/cmdline`` mention ``marker``? See module docstring."""
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    if not cmdline_path.exists():
        return None  # no /proc entry: cannot tell (or the pid is gone)
    try:
        cmdline = cmdline_path.read_bytes().decode(errors="replace")
    except OSError as exc:
        logger.debug("cannot read cmdline of pid %s: %s", pid, exc)
        return None
    return marker in cmdline