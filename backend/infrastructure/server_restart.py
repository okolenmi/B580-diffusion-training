"""ExecServerRestarter -- the ServerRestarter port by re-running the launcher.

Re-executing the launcher rather than ``backend.cli`` directly keeps one place that knows the arguments, ``.env`` and interpreter
order (``backend.first_run.relaunch``).
"""

from __future__ import annotations

import sys
import threading

from .. import first_run
from ..application.ports.server_restart import ServerRestarter

#: Long enough for the response that asked for the restart to reach the browser.
_DELAY_SECONDS = 1.5


class ExecServerRestarter(ServerRestarter):
    def running_in(self, interpreter: str) -> bool:
        return first_run.same_environment(interpreter, sys.executable)

    def can_serve(self, interpreter: str) -> bool:
        return first_run.can_serve(interpreter)

    def restart_into(self, interpreter: str) -> None:
        timer = threading.Timer(_DELAY_SECONDS, first_run.relaunch, args=(interpreter, sys.argv[1:]), kwargs={"wait": False})
        timer.daemon = True
        timer.start()


__all__ = ["ExecServerRestarter"]
