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
from backend.presentation.app import create_app
from backend.tests.support import (
    asgi_request,
    check,
    finish,
)


def _build(tmp: str) -> tuple[Container, object]:
    settings = Settings(project_root=Path(tmp), db_path=Path(tmp) / "backend.db")
    container = build_container(settings)
    return container, create_app(container.services)




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
                    if b"graph_execution_finished" in so_far:
                        disconnect.set()

            async def publisher():
                # First frame is out => the bus subscription exists.
                await first_seen.wait()
                from backend.domain.events import GraphExecutionFinished

                container.services.event_bus.publish(
                    GraphExecutionFinished(execution_id=1, nodes=5)
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
            b"graph_execution_finished" in body and b'"execution_id": 1' in body,
            "published event arrived as a data frame",
        )
        check(True, "stream shut down cleanly (no deadlock, no timeout)")






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
            app, "/api/v1/config", method="PATCH",
            json_body={"path": "cfg/missing.toml", "overrides": {}},
            extra_headers={"origin": "http://evil.example"},
        )
        check(
            status == 403 and body["error"]["code"] == "forbidden_origin",
            f"a cross-site POST is refused (got {status} {body!r})",
        )

        # Same-origin write: allowed (this is what the frontend does).
        status, _, body = asgi_request(
            app, "/api/v1/config", method="PATCH",
            json_body={"path": "cfg/missing.toml", "overrides": {}},
            extra_headers={"origin": "http://localhost"},
        )
        check(
            status == 404 and body["error"]["code"] == "config_not_found",
            f"a same-origin write reaches the handler (got {status} {body!r})",
        )

        # Reads stay open: a cross-origin GET changes nothing.
        status, _, body = asgi_request(
            app, "/api/v1/health", extra_headers={"origin": "http://evil.example"}
        )
        check(status == 200, f"cross-origin reads are allowed (got {status})")

        # No Origin at all is not a browser cross-site request.
        status, _, body = asgi_request(
            app, "/api/v1/config", method="PATCH",
            json_body={"path": "cfg/missing.toml", "overrides": {}},
        )
        check(
            status == 404,
            f"a request without Origin is not blocked (got {status})",
        )


def main() -> None:
    test_health()
    test_request_guard()
    test_sse_stream()
    finish()


if __name__ == "__main__":
    main()
