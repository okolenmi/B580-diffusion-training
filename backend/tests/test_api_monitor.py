"""Monitor stream tests -- the SSE contract for one monitor_id (M6).

Verifies the pinned frame sequence from
``docs/design/backend/03-migration-strategy.md`` section 4 end to end
against the real composition root: opener, history replay, live
reports, clear broadcast, clean shutdown on disconnect -- plus the
guarantee that page routes never shadow the API's error envelope.

Run directly: python backend/tests/test_api_monitor.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.bootstrap import Container, build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, check, finish


def _build(tmp: str) -> tuple[Container, object]:
    settings = Settings(project_root=Path(tmp), db_path=Path(tmp) / "backend.db")
    container = build_container(settings)
    return container, create_app(container.services)


def test_monitor_stream() -> None:
    print("\n== GET /api/v1/monitor/{id}/stream (SSE end-to-end) ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, app = _build(tmp)
        bus = container.services.monitor_bus

        # History exists before anyone connects: a dashboard opened
        # mid-run must get the run so far, not just future frames.
        bus.report("mon-e2e", {"step": 1, "loss": 0.5})

        async def stream() -> tuple[dict, list[bytes]]:
            start: dict = {}
            chunks: list[bytes] = []
            stage = {"n": 0}  # 0 wait connected -> 1 live -> 2 clear
            done = asyncio.Event()

            async def receive():
                # No body is ever sent; a disconnect unblocks the
                # response task only once the test is finished with it.
                if done.is_set():
                    return {"type": "http.disconnect"}
                await asyncio.sleep(3600)
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    start.update(message)
                elif message["type"] == "http.response.body":
                    chunks.append(message.get("body", b""))
                    body = b"".join(chunks)
                    if stage["n"] == 0 and b'"connected"' in body:
                        stage["n"] = 1
                        # Opener is out => the subscription exists.
                        bus.report("mon-e2e", {"step": 2, "loss": 0.4})
                    elif stage["n"] == 1 and b'"loss": 0.4' in body:
                        stage["n"] = 2
                        bus.clear("mon-e2e")
                    elif stage["n"] == 2 and b'"clear"' in body:
                        done.set()

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/api/v1/monitor/mon-e2e/stream",
                "raw_path": b"/api/v1/monitor/mon-e2e/stream",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"host", b"localhost"), (b"accept", b"text/event-stream")],
                "client": ("1.2.3.4", 1234),
                "server": ("localhost", 8766),
            }
            # A deadlock (loop capture, replay, disconnect detection)
            # surfaces here as TimeoutError instead of a silent hang.
            await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
            return start, chunks

        start, chunks = asyncio.run(stream())
        body = b"".join(chunks)
        headers = {
            k.decode().lower(): v.decode() for k, v in start.get("headers", [])
        }

        check(start.get("status") == 200, f"status 200 (got {start.get('status')})")
        check(
            headers.get("content-type", "").startswith("text/event-stream"),
            f"content-type {headers.get('content-type')!r}",
        )
        check(headers.get("cache-control") == "no-cache", "no-cache headers")
        check(
            headers.get("x-accel-buffering") == "no",
            "intermediary buffering disabled",
        )
        check(body.startswith(b'data: {"type": "connected"}'), "opener frame first")
        check(b'"step": 1' in body, "pre-connect history replayed")
        check(
            body.index(b'"step": 1') > body.index(b'"connected"'),
            "replay follows the opener",
        )
        check(b'"step": 2' in body, "live report arrived")
        check(b'"type": "clear"' in body, "clear frame broadcast")
        check(True, "stream shut down cleanly on disconnect (no deadlock)")


def test_api_miss_keeps_error_envelope() -> None:
    print("\n== pages never shadow the API error envelope ==")
    with tempfile.TemporaryDirectory() as tmp:
        _, app = _build(tmp)
        # No static_dir wired: page paths must still answer with the
        # JSON envelope, never static 404 HTML.
        status, _, body = asgi_request(app, "/")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"/ answers 404 in the error envelope (got {status} {body!r})",
        )
        status, _, body = asgi_request(app, "/api/v1/nope")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"unknown API route stays in the envelope (got {status} {body!r})",
        )


def test_subscriber_backlog_is_bounded() -> None:
    # docs 07 F-14: a subscriber that stops consuming must not grow the
    # process's memory forever. Telemetry frames are dropped oldest-first;
    # the state frames (clear / run_end) are kept.
    print("\n== monitor bus: a stalled subscriber's queue stays bounded ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, _ = _build(tmp)
        bus = container.services.monitor_bus

        async def scenario() -> None:
            queue = bus.subscribe("mon-bound")
            for step in range(1, 1500):
                bus.report("mon-bound", {"step": step, "loss": 0.5})
            await asyncio.sleep(0)  # let the loop-drained callbacks land
            check(
                queue.qsize() <= 512,
                f"the backlog is capped (got {queue.qsize()} frames)",
            )
            frames = [queue.get_nowait() for _ in range(queue.qsize())]
            check(
                any('"step": 1499' in frame for frame in frames),
                "the newest report is still in there",
            )
            check(
                not any('"step": 1,' in frame for frame in frames),
                "and the oldest ones are what got dropped",
            )
            bus.report("mon-bound", {"type": "run_end", "step": 1499})
            await asyncio.sleep(0)
            frames = [queue.get_nowait() for _ in range(queue.qsize())]
            check(
                any('"type": "run_end"' in frame for frame in frames),
                "the state frame survived the overflow",
            )
            bus.unsubscribe("mon-bound", queue)

        asyncio.run(scenario())


test_subscriber_backlog_is_bounded()
finish()
