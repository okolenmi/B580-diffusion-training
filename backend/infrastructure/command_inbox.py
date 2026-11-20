"""The command inbox: how a page's action reaches a run that lives in another process.

A run's child process outlives a server restart (ADR 0003), so a pipe to it would be lost exactly when it matters. Instead each
run has a folder next to its event file -- ``execution_<N>.commands/`` -- and the two sides meet there, mirroring the event file in
the other direction:

* ``CommandOutbox`` (server) writes one ``<time>-<random>.json`` per command, atomically (temp file, then rename), so the reader never
  sees half a file.
* ``CommandInbox`` (child) is a daemon thread that lists the folder in name order, hands each command to ``ActionDispatcher.dispatch``
  and deletes the file. Answers come back the way every report does: as frames in the event file.

The two share only the file format, like the event stream; a command is ``{"id", "node", "action"}`` and nothing else.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_NAME = re.compile(r"^\d{20}-[0-9a-f]{8}\.json$")
POLL_INTERVAL_S = 0.2


def commands_dir_for(scratch_dir: Path, execution_id: int) -> Path:
    return scratch_dir / f"execution_{execution_id}.commands"


class CommandOutbox:
    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def submit(self, node: str, action: str) -> str:
        """Queue one command and return its id (what the answer frames carry)."""
        self._directory.mkdir(parents=True, exist_ok=True)
        stamp, token = f"{time.time_ns():020d}", uuid.uuid4().hex[:8]
        command = {"id": f"{stamp}-{token}", "node": node, "action": action}
        temporary = self._directory / f".{stamp}-{token}.tmp"
        temporary.write_text(json.dumps(command), encoding="utf-8")
        os.replace(temporary, self._directory / f"{stamp}-{token}.json")
        return command["id"]


class CommandInbox:
    """Reads a run's command folder until stopped. ``dispatch`` runs on this thread; it must only queue work."""

    def __init__(self, directory: Path, dispatch: Callable[[dict], None], interval_s: float = POLL_INTERVAL_S) -> None:
        self._directory = directory
        self._dispatch = dispatch
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="backend-graph-command-inbox", daemon=True)

    def start(self) -> None:
        self._directory.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def __enter__(self) -> CommandInbox:
        self.start()
        return self

    def __exit__(self, *exc) -> bool:
        self.stop()
        return False

    def poll(self) -> int:
        """Take every waiting command now (the loop's body; tests call it directly)."""
        taken = 0
        try:
            names = sorted(n for n in os.listdir(self._directory) if _NAME.match(n))
        except FileNotFoundError:
            return 0
        for name in names:
            path = self._directory / name
            try:
                command = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("unreadable command file %s dropped", path)
                command = None
            try:
                path.unlink()
            except OSError:
                pass
            if isinstance(command, dict):
                try:
                    self._dispatch(command)
                except Exception:  # noqa: BLE001 -- one bad command must not end the inbox
                    logger.exception("dispatching command %s failed", name)
                taken += 1
        return taken

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            self.poll()
