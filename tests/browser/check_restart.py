"""A server that goes away while a run trains (TESTING 99.3 and the adoption path): the run is a child process of its own, so a clean stop, a crash and a
power cut of the whole machine must each leave the history true and the card free. A real server and real children on tiny CPU stand-ins.

    python -m tests.browser.check_restart
"""

from __future__ import annotations

import os
import signal
import time

from tests.support import check
from . import fixtures
from .check_runs import executions, idle, stop_all
from .server import TestServer
from .session import Chromium


def start_run(server, dataset: str, steps: int, name: str) -> int:
    graph = fixtures.lora_graph(server.home, dataset, steps=steps, save_every=1000, out=name)
    answer = server.api("POST", "/api/v1/graphs/run", {"graph": graph, "mode": "auto"})
    run_id = answer.get("execution_id") or answer.get("id")
    server.wait_for("the run to train", lambda: _status(server, run_id) == "running", 60)
    return run_id


def _status(server, run_id: int) -> str:
    return next((e["status"] for e in executions(server) if e["execution_id"] == run_id), "missing")


def survives(server, dataset, how: str, sig) -> None:
    run_id = start_run(server, dataset, 400, f"keep-{how}")
    kids = server.children()
    server.restart(keep_runs=True, sig=sig)
    check(bool(kids) and all(_alive(k) for k in kids), f"{how}: the child is still alive after the server went away")
    check(_status(server, run_id) == "running", f"{how}: the new server shows the same run as running ({_status(server, run_id)})")
    stop_now = server.api("POST", f"/api/v1/graphs/executions/{run_id}/stop")
    server.wait_for("the adopted run to stop", lambda: _status(server, run_id) in ("stopped", "finished", "error"), 60)
    check(_status(server, run_id) == "stopped", f"{how}: Stop on an adopted run ends it as stopped ({_status(server, run_id)})")
    idle(server)
    again = start_run(server, dataset, 4, f"after-{how}")
    server.wait_for("the next run to end", lambda: _status(server, again) == "finished", 60)
    check(True, f"{how}: the card is free for the next run")


def finishes_while_down(server, dataset) -> None:
    run_id = start_run(server, dataset, 30, "while-down")
    server.stop(keep_runs=True)
    server.wait_for("the child to end by itself", lambda: not server.children(), 90)
    server.start()
    server.wait_for("the history to catch up", lambda: _status(server, run_id) in ("finished", "stopped", "error"), 30)
    check(_status(server, run_id) == "finished", f"a run that ended while the server was down is recorded as finished ({_status(server, run_id)})")
    idle(server, 10)


def machine_loses_power(server, dataset) -> None:
    run_id = start_run(server, dataset, 400, "power")
    server.stop(keep_runs=True, sig=signal.SIGKILL)
    server._kill_graph_children()
    server.start()
    server.wait_for("the history to catch up", lambda: _status(server, run_id) not in ("running", "queued", "planned"), 40)
    check(_status(server, run_id) in ("error", "stopped"), f"a run whose child died with the server does not stay 'running' ({_status(server, run_id)})")
    again = start_run(server, dataset, 4, "after-power")
    server.wait_for("the next run to end", lambda: _status(server, again) == "finished", 60)
    check(True, "and a new run can start")


def child_killed(server, dataset, browser) -> None:
    """A child the kernel kills (the out-of-memory killer sends SIGKILL): the row says how it died, the log is reachable, the start page shows both."""
    run_id = start_run(server, dataset, 400, "killed")
    for pid in server.children():
        os.kill(pid, signal.SIGKILL)
    server.wait_for("the row to turn terminal", lambda: _status(server, run_id) not in ("running", "queued", "planned"), 40)
    row = next(e for e in executions(server) if e["execution_id"] == run_id)
    check(row["status"] == "error" and "signal 9 (SIGKILL)" in row["error"] and "out-of-memory" in row["error"], f"the row names the signal and what it usually means ({row['error'][:160]!r})")
    log = server.http("GET", f"/api/v1/graphs/executions/{run_id}/log")
    check(log[0] == 200 and "text/plain" in log[1], f"the run's log is served as text ({log[0]} {log[1]})")
    page = browser.new_page()
    page.goto("/", settle_ms=1500)
    page.page.locator(".run-more summary").first.click()
    check(page.visible(".run-error") and "SIGKILL" in page.text(".run-error") and page.visible(".run-error-links a"), "the start page opens the error and a link to the full log from the failed run")
    check(server.http("GET", "/api/v1/graphs/executions/9999/log")[0] == 404, "a log that does not exist is a 404")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main() -> None:
    with TestServer() as server, Chromium(server.url) as browser:
        dataset = fixtures.toy_dataset(server.home)
        for name, part in (("clean stop", lambda: survives(server, dataset, "clean stop", signal.SIGTERM)),
                           ("crash", lambda: survives(server, dataset, "crash", signal.SIGKILL)),
                           ("finishes while down", lambda: finishes_while_down(server, dataset)),
                           ("power cut", lambda: machine_loses_power(server, dataset)),
                           ("killed child", lambda: child_killed(server, dataset, browser))):
            print(f"\n{name}")
            try:
                part()
            except Exception as error:  # noqa: BLE001
                check(False, f"{name} stopped: {type(error).__name__}: {str(error)[:400]}")
            try:
                stop_all(server)
            except Exception:  # noqa: BLE001
                pass


if __name__ == "__main__":
    from tests.support import finish
    main()
    finish()
