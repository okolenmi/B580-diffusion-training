"""Cache-Control policy for browser-facing responses (server/main.py).

Why this exists: without a Cache-Control header the browser applies
heuristic freshness, so a normal reload can serve yesterday's JS against
today's server introspect data -- a real mixed-version bug hit once
(old nodegraph.js's strict-equality visible_when check vs the new
array-valued gates: every gated row silently hidden until a hard
reload). The fix is a single HTTP middleware on the app that stamps
`Cache-Control: no-cache` on /static assets and on HTML pages (no-cache
revalidates, it does not disable caching -- 304 when unchanged).

What's checked:

- /static assets and route-served HTML carry `Cache-Control: no-cache`.
- API/JSON responses and the /datasets + /runs data mounts are NOT
  forced to revalidate (their staleness is a preview nit, not a
  correctness bug -- and image revalidation would be pure chattiness).
- This app's first middleware is a BaseHTTPMiddleware -- the classic
  deadlock/footgun zone for Server-Sent Events. The monitor dashboard
  streams from /api/sse, so the SSE endpoint is exercised through the
  real app: the immediate "connected" event must arrive, and a client
  disconnect must cancel the generator cleanly, all under a hard
  timeout (a middleware/SSE deadlock hangs instead of failing).

Runs the real FastAPI app in-process over raw ASGI -- no server, no
GPU, no browser. Run directly:
`python server/smoke_tests/smoke_test_static_caching.py`
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server.main import app  # noqa: E402 -- imports after sys.path, as everywhere here

failures = []


def check(condition: bool, message: str):
    print(f"  {'PASS' if condition else 'FAIL'}: {message}")
    if not condition:
        failures.append(message)


async def _get(path: str, query: str = "") -> tuple[int | None, dict[str, str], bytes]:
    """One GET through the whole app (middleware included); returns
    (status, lowercased headers, body) -- body only what was sent before
    disconnecting (for streaming endpoints)."""
    start, chunks = {}, []
    disconnected = asyncio.Event()

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            start.update(msg)
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))
            disconnected.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
             "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": query.encode(),
             "root_path": "", "headers": [(b"host", b"localhost")],
             "client": ("1.2.3.4", 1), "server": ("localhost", 8000)}
    await app(scope, receive, send)
    headers = {k.decode().lower(): v.decode() for k, v in start.get("headers", [])}
    return start.get("status"), headers, b"".join(chunks)


def run(coro):
    return asyncio.run(coro)


def check_browser_assets_revalidate():
    print("\n== /static assets and HTML pages revalidate on every reload ==")
    for path in ("/static/nodegraph.js", "/static/style.css", "/static/monitor_dashboard.js"):
        status, headers, _ = run(_get(path))
        check(status == 200 and headers.get("cache-control") == "no-cache",
              f"{path} -> {status}, cache-control={headers.get('cache-control')!r} "
              f"(want 200, 'no-cache')")
    for path in ("/", "/nodegraph"):
        status, headers, _ = run(_get(path))
        check(status == 200 and headers.get("cache-control") == "no-cache",
              f"{path} -> {status}, cache-control={headers.get('cache-control')!r} "
              f"(want 200, 'no-cache')")


def check_data_paths_left_alone():
    print("\n== API/JSON and data mounts keep default caching ==")
    for path in ("/api/definitely-not-a-route", "/runs/no-such-preview.png"):
        status, headers, _ = run(_get(path))
        check(status == 404 and "cache-control" not in headers,
              f"{path} -> {status}, cache-control absent "
              f"(want 404, header not stamped)")


def check_sse_survives_the_middleware():
    print("\n== /api/sse still streams through the app's first middleware ==")
    status, headers, body = run(asyncio.wait_for(
        _get("/api/sse", "run_id=0"), timeout=5.0))
    check(status == 200, f"SSE status {status} (want 200)")
    check(headers.get("content-type", "").startswith("text/event-stream"),
          f"SSE content-type {headers.get('content-type')!r} "
          f"(want text/event-stream)")
    check(b"connected" in body,
          "the immediate 'connected' event arrived before disconnect "
          "(a middleware/SSE deadlock hangs here instead -- the 5s timeout "
          "is what would fire)")


def main():
    check_browser_assets_revalidate()
    check_data_paths_left_alone()
    check_sse_survives_the_middleware()
    print()
    print("=" * 60)
    if failures:
        print(f"SMOKE TEST: {len(failures)} FAILURE(S)")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
