"""The first-run browser opens only once the server answers.

The bootstrap restarts the server and used to open the page at once, so the browser showed "unable to connect" until a manual
reload. The waiting is tested against a real local socket that starts listening late; the opener is the detached process's entry.

Run directly: python backend/tests/test_first_run_browser.py
"""

from __future__ import annotations

import http.server
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend import first_run
from backend.tests.support import check, finish


class _Quiet(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_waits_for_a_server_that_starts_late() -> None:
    port = _free_port()
    server_holder: list[http.server.HTTPServer] = []

    def start_later() -> None:
        time.sleep(1.0)
        server = http.server.HTTPServer(("127.0.0.1", port), _Quiet)
        server_holder.append(server)
        server.serve_forever()

    threading.Thread(target=start_later, daemon=True).start()
    began = time.monotonic()
    ok = first_run.wait_until_serving(f"http://127.0.0.1:{port}/setup", timeout=15, interval=0.1)
    check(ok and time.monotonic() - began >= 0.9, "returns true only after the server accepts connections")
    for server in server_holder:
        server.shutdown()


def test_gives_up_after_the_timeout() -> None:
    now = [0.0]
    ok = first_run.wait_until_serving(f"http://127.0.0.1:{_free_port()}/", timeout=5, interval=1,
                                      sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])
    check(ok is False and now[0] >= 5, "a server that never comes up ends the wait")


def test_opens_the_browser_only_when_serving() -> None:
    opened: list[str] = []
    original = (first_run.wait_until_serving, first_run.open_browser)
    try:
        first_run.open_browser = lambda url: opened.append(url) or True
        first_run.wait_until_serving = lambda url, **kw: False
        first_run.open_when_serving("http://localhost:1/setup")
        check(opened == [], "no browser when the server never answers")
        first_run.wait_until_serving = lambda url, **kw: True
        first_run.open_when_serving("http://localhost:1/setup")
        check(opened == ["http://localhost:1/setup"], "the browser opens once it answers")
    finally:
        first_run.wait_until_serving, first_run.open_browser = original


def test_the_helper_entry_does_not_install_anything() -> None:
    original = first_run.open_when_serving
    seen: list[str] = []
    try:
        first_run.open_when_serving = lambda url: seen.append(url) or True
        code = first_run.main(["--open-when-serving", "http://localhost:1/setup"])
    finally:
        first_run.open_when_serving = original
    check(code == 0 and seen == ["http://localhost:1/setup"], "--open-when-serving only waits and opens")


if __name__ == "__main__":
    test_waits_for_a_server_that_starts_late()
    test_gives_up_after_the_timeout()
    test_opens_the_browser_only_when_serving()
    test_the_helper_entry_does_not_install_anything()
    finish()
