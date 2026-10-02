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
from backend.infrastructure.persistence.run_repository import SqliteRunRepository
from backend.presentation.app import create_app
from backend.tests.support import (
    FakeClock,
    FakeTrainingGateway,
    asgi_request,
    build_services,
    check,
    finish,
)


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

        status, _, body = asgi_request(app, "/api/v1/runs", method="PUT")
        check(
            status == 405 and body["error"]["code"] == "http_405",
            f"method not allowed uses the envelope too (got {status} {body!r})",
        )


def test_diverged_run_body() -> None:
    # docs 07 F-03: a non-finite loss in the row used to raise inside
    # Starlette's allow_nan=False response and 500 the whole page.
    # SQLite maps NaN to NULL, so +/-Inf is what survives the round
    # trip into the DTO (NaN only ever reaches the SSE frame, pinned
    # in test_json_safe.py).
    print("\n== GET /api/v1/runs with a diverged (inf) loss ==")
    with tempfile.TemporaryDirectory() as tmp:
        container, app = _build(tmp)
        repo = SqliteRunRepository(container.database)
        clock = FakeClock()
        run = Run.create(
            config_path="configs/diverged.toml",
            mode="distillation",
            total_steps=100,
            created_at=clock.now(),
        )
        repo.add(run)
        run.mark_started(pid=4242, at=clock.now())
        repo.update(run)
        run.record_progress(
            done_steps=50, at=clock.now(),
            current_loss=float("inf"), avg_loss=float("-inf"), phase="training",
        )
        repo.update(run)

        status, _, body = asgi_request(app, "/api/v1/runs/1", strict_json=True)
        check(status == 200, f"diverged run still serves (got {status})")
        check(isinstance(body, dict), f"body is strict JSON (got {body!r})")
        check(
            body["current_loss"] is None and body["avg_loss"] is None,
            f"non-finite floats are null (got {body['current_loss']!r})",
        )
        check(
            body["nonfinite"] == {"current_loss": "inf", "avg_loss": "-inf"},
            f"the UI can render the diverged state (got {body.get('nonfinite')})",
        )
        check(body["done_steps"] == 50, "the finite fields are untouched")

        status, _, listing = asgi_request(app, "/api/v1/runs", strict_json=True)
        check(
            status == 200 and listing["runs"][0]["nonfinite"]["current_loss"] == "inf",
            "the history page survives a diverged run too",
        )
        status, _, active = asgi_request(app, "/api/v1/runs/active", strict_json=True)
        check(
            status == 200 and active["nonfinite"]["current_loss"] == "inf",
            "so does the active-run read the dashboard polls",
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
            and isinstance(published[0].event, RunsDeleted)
            and published[0].event.deleted == 2,
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


def _build_faked(tmp: str) -> tuple[object, object]:
    """App over in-memory services + fake gateway (no real spawning)."""
    project = Path(tmp)
    (project / "cfg.toml").write_text("[common]\nsteps = 100\n", encoding="utf-8")
    gateway = FakeTrainingGateway()
    services = build_services(
        gateway=gateway,
        project_root=project,
        runs_dir=project / "runs",
        poll_interval=0.02,
    )
    return gateway, create_app(services)


def test_lifecycle_endpoints() -> None:
    print("\n== POST /runs, /stop, /active, /log (faked gateway) ==")
    with tempfile.TemporaryDirectory() as tmp:
        gateway, app = _build_faked(tmp)

        status, _, body = asgi_request(app, "/api/v1/runs/active")
        check(
            status == 404 and body["error"]["code"] == "no_active_run",
            f"no active run -> 404 no_active_run (got {status} {body!r})",
        )

        # Launch-validation failures first, while no run is active yet
        # (the active check would 409 before config validation).
        status, _, body = asgi_request(
            app,
            "/api/v1/runs",
            method="POST",
            json_body={"config_path": "cfg.toml", "start_from": "nope"},
        )
        check(
            status == 422 and body["error"]["code"] == "invalid_query",
            f"bad start_from -> 422 invalid_query (got {status} {body!r})",
        )

        status, _, body = asgi_request(
            app, "/api/v1/runs", method="POST", json_body={"config_path": "gone.toml"}
        )
        check(
            status == 404 and body["error"]["code"] == "config_not_found",
            f"missing config -> 404 config_not_found (got {status} {body!r})",
        )

        status, _, body = asgi_request(app, "/api/v1/runs", method="POST")
        check(
            status == 422 and body["error"]["code"] == "validation_error",
            f"no body -> 422 validation_error (got {status} {body!r})",
        )

        status, _, body = asgi_request(
            app, "/api/v1/runs", method="POST", json_body={"config_path": "cfg.toml"}
        )
        check(status == 201, f"start -> 201 (got {status} {body!r})")
        check(
            body["id"] == 1 and body["status"] == "running" and body["pid"] is not None,
            "launched run returned with pid",
        )
        check(len(gateway.spawned) == 1, "fake gateway received the spawn")

        status, _, body = asgi_request(
            app, "/api/v1/runs", method="POST", json_body={"config_path": "cfg.toml"}
        )
        check(
            status == 409 and body["error"]["code"] == "run_already_active",
            f"second start -> 409 run_already_active (got {status} {body!r})",
        )

        status, _, body = asgi_request(app, "/api/v1/runs/active")
        check(status == 200 and body["id"] == 1, "active run visible after start")

        log_path = Path(tmp) / "runs" / "run_1" / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("a\nb\nc\n", encoding="utf-8")
        status, _, body = asgi_request(app, "/api/v1/runs/1/log?lines=2")
        check(
            status == 200 and body == {"log": "b\nc\n"},
            f"log tail (got {status} {body!r})",
        )
        status, _, body = asgi_request(app, "/api/v1/runs/1/log?lines=999")
        check(
            status == 422 and body["error"]["code"] == "invalid_query",
            f"lines out of range -> 422 (got {status} {body!r})",
        )

        status, _, body = asgi_request(
            app, "/api/v1/runs/1/stop", method="POST", json_body={"force": True}
        )
        check(status == 200 and body["status"] == "cancelled", "stop -> 200 cancelled")
        check(gateway.stopped == [(4242, True)], "force forwarded to the gateway")

        status, _, body = asgi_request(app, "/api/v1/runs/1/stop", method="POST")
        check(
            status == 409 and body["error"]["code"] == "run_not_running",
            f"second stop -> 409 run_not_running (got {status} {body!r})",
        )

        status, _, body = asgi_request(app, "/api/v1/runs/active")
        check(
            status == 404 and body["error"]["code"] == "no_active_run",
            "cancelled run no longer active",
        )


def test_request_guard() -> None:
    # docs 07 F-06: no auth is a documented decision, but DNS rebinding
    # and cross-site writes are not something "bound to loopback" already
    # prevents -- a browser will happily send both.
    print("\n== Host / Origin guard (rebinding + cross-site writes) ==")
    with tempfile.TemporaryDirectory() as tmp:
        _, app = _build(tmp)

        status, _, body = asgi_request(
            app, "/api/v1/health", extra_headers={"host": "evil.example"}
        )
        check(
            status == 403 and body["error"]["code"] == "forbidden_host",
            f"a foreign Host is refused (got {status} {body!r})",
        )
        status, _, body = asgi_request(
            app, "/api/v1/health", extra_headers={"host": "127.0.0.1:8766"}
        )
        check(status == 200, f"our own host with a port is fine (got {status})")
        status, _, body = asgi_request(app, "/api/v1/health")
        check(status == 200, f"localhost is fine (got {status})")

        # Cross-site write: refused, whatever the body was going to be.
        status, _, body = asgi_request(
            app, "/api/v1/runs", method="POST",
            json_body={"config_path": "configs/distill.toml"},
            extra_headers={"origin": "http://evil.example"},
        )
        check(
            status == 403 and body["error"]["code"] == "forbidden_origin",
            f"a cross-site POST is refused (got {status} {body!r})",
        )

        # Same-origin write: allowed (this is what the frontend does).
        status, _, body = asgi_request(
            app, "/api/v1/runs", method="POST",
            json_body={"config_path": "configs/missing.toml"},
            extra_headers={"origin": "http://localhost"},
        )
        check(
            status == 404 and body["error"]["code"] == "config_not_found",
            f"a same-origin POST reaches the handler (got {status} {body!r})",
        )

        # Reads stay open: a cross-origin GET changes nothing.
        status, _, body = asgi_request(
            app, "/api/v1/health", extra_headers={"origin": "http://evil.example"}
        )
        check(status == 200, f"cross-origin reads are allowed (got {status})")

        # No Origin at all is not a browser cross-site request.
        status, _, body = asgi_request(
            app, "/api/v1/runs", method="POST",
            json_body={"config_path": "configs/missing.toml"},
        )
        check(
            status == 404,
            f"a request without Origin is not blocked (got {status})",
        )


def main() -> None:
    test_health()
    test_list_runs()
    test_get_run_and_errors()
    test_diverged_run_body()
    test_request_guard()
    test_delete_runs()
    test_lifecycle_endpoints()
    test_sse_stream()
    finish()


if __name__ == "__main__":
    main()
