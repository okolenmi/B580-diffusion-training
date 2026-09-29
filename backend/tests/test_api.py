"""End-to-end tests -- the whole stack over raw ASGI (temp database,
real wiring from ``bootstrap.build_container``, fakes only where a
temp directory is not required).

Run directly: python backend/tests/test_api.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.bootstrap import Container, build_container
from backend.config import Settings
from backend.domain.entities.run import Run
from backend.domain.events import RunsDeleted
from backend.domain.value_objects import RunStatus
from backend.infrastructure.persistence.run_repository import SqliteRunRepository
from backend.presentation.app import create_app
from backend.tests.support import FakeClock, asgi_request, check, finish


def _build(tmp: str) -> tuple[Container, object]:
    settings = Settings(project_root=Path(tmp), db_path=Path(tmp) / "backend.db")
    container = build_container(settings)
    return container, create_app(container.services)


def _seed(container: Container) -> None:
    repo = SqliteRunRepository(container.database)
    clock = FakeClock()
    created = Run.create(
        config_path="configs/first.toml",
        mode="distillation",
        total_steps=100,
        created_at=clock.now(),
    )
    repo.add(created)
    clock.advance(2)
    running = Run.create(
        config_path="configs/second.toml",
        mode="lora",
        total_steps=200,
        created_at=clock.now(),
    )
    repo.add(running)
    running.mark_started(pid=777, at=clock.now())
    repo.update(running)


def test_health() -> None:
    print("\n== GET /api/v1/health ==")
    with tempfile.TemporaryDirectory() as tmp:
        _, app = _build(tmp)
        status, _, body = asgi_request(app, "/api/v1/health")
        check(status == 200, f"status 200 (got {status})")
        check(
            isinstance(body, dict)
            and body.get("status") == "ok"
            and str(body.get("version", "")).startswith("0."),
            f"health payload {body!r}",
        )


def test_list_runs() -> None:
    print("\n== GET /api/v1/runs (shape, order, filters) ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, app = _build(tmp)
        _seed(container)

        status, headers, body = asgi_request(app, "/api/v1/runs")
        check(status == 200, f"status 200 (got {status})")
        check(
            headers.get("content-type", "").startswith("application/json"),
            "JSON content type",
        )
        check(body["count"] == 2 and len(body["runs"]) == 2, "both seeded runs listed")
        check(
            [r["id"] for r in body["runs"]] == [2, 1],
            f"newest first (got {[r['id'] for r in body['runs']]})",
        )
        check(body["runs"][0]["status"] == "running", "status rendered as string value")
        check(body["runs"][0]["pid"] == 777, "pid roundtrips through all layers")
        check(body["runs"][0]["config_path"] == "configs/second.toml", "fields intact")
        expected_keys = {"id", "status", "config_path", "mode", "total_steps", "created_at"}
        check(
            expected_keys <= set(body["runs"][0]),
            "response carries the documented run fields",
        )

        _, _, paged = asgi_request(app, "/api/v1/runs?limit=1")
        check(paged["count"] == 1 and paged["runs"][0]["id"] == 2, "?limit=1 pages")

        _, _, only_running = asgi_request(app, "/api/v1/runs?status=running")
        check(
            only_running["count"] == 1 and only_running["runs"][0]["id"] == 2,
            "?status=running filters",
        )

        status, _, body = asgi_request(app, "/api/v1/runs?status=bogus")
        check(
            status == 422
            and isinstance(body, dict)
            and body["error"]["code"] == "invalid_query"
            and "bogus" in body["error"]["message"],
            f"unknown status -> 422 invalid_query envelope (got {status} {body!r})",
        )

        status, _, body = asgi_request(app, "/api/v1/runs?limit=0")
        check(
            status == 422 and body["error"]["code"] == "invalid_query",
            "limit=0 -> 422 invalid_query (use case validates, not Query())",
        )

        status, _, body = asgi_request(app, "/api/v1/runs?limit=abc")
        check(
            status == 422 and body["error"]["code"] == "validation_error",
            "non-integer limit -> 422 validation_error envelope",
        )


def test_get_run_and_errors() -> None:
    print("\n== GET /api/v1/runs/{id} + error envelopes ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, app = _build(tmp)
        _seed(container)

        status, _, body = asgi_request(app, "/api/v1/runs/1")
        check(status == 200 and body["id"] == 1, "existing run fetched")
        check(body["mode"] == "distillation", "DTO -> schema field mapping intact")

        status, _, body = asgi_request(app, "/api/v1/runs/999")
        check(
            status == 404
            and body["error"]["code"] == "run_not_found"
            and "999" in body["error"]["message"],
            f"missing run -> 404 run_not_found (got {status} {body!r})",
        )

        status, _, body = asgi_request(app, "/api/v1/nope")
        check(
            status == 404
            and isinstance(body, dict)
            and body["error"]["code"] == "http_404",
            f"unknown route -> same envelope, no bare detail (got {body!r})",
        )

        status, _, body = asgi_request(app, "/api/v1/runs", method="POST")
        check(
            status == 405 and body["error"]["code"] == "http_405",
            f"method not allowed uses the envelope too (got {status} {body!r})",
        )


def test_delete_runs() -> None:
    print("\n== DELETE /api/v1/runs (mutation + event) ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, app = _build(tmp)
        _seed(container)
        published: list[object] = []
        container.services.event_bus.subscribe(published.append)

        status, _, body = asgi_request(app, "/api/v1/runs", method="DELETE")
        check(status == 200 and body == {"deleted": 2}, f"deleted 2 (got {status} {body!r})")
        check(
            len(published) == 1
            and isinstance(published[0], RunsDeleted)
            and published[0].deleted == 2,
            "RunsDeleted event published on the real bus",
        )

        _, _, after = asgi_request(app, "/api/v1/runs")
        check(after["count"] == 0, "history empty after deletion")


def test_sse_stream() -> None:
    print("\n== GET /api/v1/events (SSE end-to-end) ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, app = _build(tmp)

        async def stream() -> tuple[dict, list[bytes]]:
            start: dict = {}
            chunks: list[bytes] = []
            first_seen = asyncio.Event()
            disconnect = asyncio.Event()

            async def receive():
                if disconnect.is_set():
                    return {"type": "http.disconnect"}
                await asyncio.sleep(3600)  # cancellable; the request's
                # is_disconnected() peek runs it inside a cancelled scope
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    start.update(message)
                elif message["type"] == "http.response.body":
                    chunks.append(message.get("body", b""))
                    so_far = b"".join(chunks)
                    if b"stream_opened" in so_far:
                        first_seen.set()
                    if b"run_completed" in so_far:
                        disconnect.set()

            async def publisher():
                # First frame is out => the bus subscription exists.
                await first_seen.wait()
                from backend.domain.events import RunCompleted

                container.services.event_bus.publish(
                    RunCompleted(run_id=1, done_steps=5)
                )

            scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/api/v1/events",
                "raw_path": b"/api/v1/events",
                "query_string": b"",
                "root_path": "",
                "headers": [(b"host", b"localhost"), (b"accept", b"text/event-stream")],
                "client": ("1.2.3.4", 1234),
                "server": ("localhost", 8766),
            }
            pub = asyncio.create_task(publisher())
            # A deadlock (loop hop, subscription, disconnect detection)
            # surfaces here as TimeoutError instead of a silent hang.
            await asyncio.wait_for(app(scope, receive, send), timeout=5.0)
            pub.cancel()
            return start, chunks

        start, chunks = asyncio.run(stream())
        body = b"".join(chunks)
        headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}

        check(start.get("status") == 200, f"stream status 200 (got {start.get('status')})")
        check(
            headers.get("content-type", "").startswith("text/event-stream"),
            f"content-type {headers.get('content-type')!r}",
        )
        check(
            headers.get("cache-control") == "no-cache",
            "stream tells proxies not to cache",
        )
        check(b"stream_opened" in body, "first frame is stream_opened")
        check(
            b"run_completed" in body and b'"run_id": 1' in body,
            "published event arrived as a data frame",
        )
        check(True, "stream shut down cleanly (no deadlock, no timeout)")


def main() -> None:
    test_health()
    test_list_runs()
    test_get_run_and_errors()
    test_delete_runs()
    test_sse_stream()
    finish()


if __name__ == "__main__":
    main()
