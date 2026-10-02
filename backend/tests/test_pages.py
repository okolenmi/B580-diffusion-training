"""Frontend serving tests -- page routes + /ui asset mount (M6--M8).

Pins register_frontend's contract: pages serve when ``static_dir`` is
given, every asset the shipped pages reference resolves, the mount
can never shadow the API's error envelope (02-api-reference.md section 1),
and page/asset responses carry ``Cache-Control: no-cache`` so a plain
reload never serves stale JS (port of server/main.py's fix).

Run directly: python backend/tests/test_pages.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.bootstrap import build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, check, finish

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"

# Every asset the shipped pages reference.
ASSETS = (
    "/ui/css/style.css",
    "/ui/css/shell.css",
    "/ui/css/monitor.css",
    "/ui/css/editor.css",
    "/ui/css/training.css",
    "/ui/css/config.css",
    "/ui/css/run.css",
    "/ui/css/datasets.css",
    "/ui/css/help.css",
    "/ui/css/settings.css",
    "/ui/js/shell.js",
    "/ui/js/api.js",
    "/ui/js/monitor.js",
    "/ui/js/lib/loss_chart.js",
    "/ui/js/views/config.js",
    "/ui/js/views/datasets.js",
    "/ui/js/editor.js",
    "/ui/js/editor/state.js",
    "/ui/js/editor/canvas.js",
    "/ui/js/editor/inspector.js",
    "/ui/js/editor/palette.js",
    "/ui/js/editor/executions.js",
    "/ui/js/editor/library.js",
)


#: FastAPI's own documentation routes. They are GET routes on the app
#: with no HTML behind them, and they are part of FastAPI's public surface
#: rather than this app's, so they are named rather than pattern-matched.
_BUILTIN_PATHS = frozenset({"/docs", "/redoc", "/openapi.json",
                            "/docs/oauth2-redirect"})


def _page_routes(app) -> dict[str, str]:
    """``{route path: html stem}`` for the page routes ``register_frontend`` added.

    Derived from the live app so a route cannot be added, or removed, without
    this noticing -- which is the whole point, since the bug it exists for
    was a route outliving the file it served.
    """
    found: dict[str, str] = {}
    for route in app.routes:
        path = getattr(route, "path", "")
        # ``include_in_schema is False`` is the marker register_frontend
        # puts on exactly the page routes: API routes carry True, and the
        # static Mount has no such attribute. Skipping everything else also
        # keeps /ui from being read as a page called "ui".
        if getattr(route, "include_in_schema", None) is not False:
            continue
        if path in _BUILTIN_PATHS or path.startswith("/api"):
            continue
        segments = [s for s in path.strip("/").split("/") if s and "{" not in s]
        found[path] = segments[-1] if segments else "index"
    return found


def _container(tmp: str):
    settings = Settings(project_root=Path(tmp), db_path=Path(tmp) / "backend.db")
    return build_container(settings)


def test_pages_and_assets() -> None:
    print("\n== pages + assets (create_app with static_dir) ==")
    check(FRONTEND.is_dir(), f"frontend directory exists ({FRONTEND})")
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_container(tmp).services, static_dir=FRONTEND)

        status, headers, body = asgi_request(app, "/")
        check(
            status == 200 and isinstance(body, str) and "<title>" in body,
            f"/ serves the app shell (got {status})",
        )
        check(
            headers.get("content-type", "").startswith("text/html"),
            f"/ content-type html (got {headers.get('content-type')!r})",
        )

        status, _, body = asgi_request(app, "/monitor/mon-test")
        check(
            status == 200 and isinstance(body, str) and "mon-loss-chart" in body,
            f"/monitor/{id} serves the monitor page (got {status})",
        )

        status, _, body = asgi_request(app, "/graph")
        check(
            status == 200 and isinstance(body, str) and "graph-canvas" in body,
            f"/graph serves the editor page (got {status})",
        )

        status, _, body = asgi_request(app, "/config")
        check(
            status == 200 and isinstance(body, str) and "config-form" in body,
            f"/config serves the config editor page (got {status})",
        )

        status, _, body = asgi_request(app, "/datasets")
        check(
            status == 200 and isinstance(body, str) and "ds-grid" in body,
            f"/datasets serves the dataset manager page (got {status})",
        )

        status, _, body = asgi_request(app, "/datasets/test2")
        check(
            status == 200 and isinstance(body, str) and "ds-grid" in body,
            f"/datasets/{{name}} serves the same detail page (got {status})",
        )

        status, _, body = asgi_request(app, "/help")
        check(
            status == 200 and isinstance(body, str) and "help-where" in body,
            f"/help serves the help page (got {status})",
        )

        status, _, body = asgi_request(app, "/settings")
        check(
            status == 200 and isinstance(body, str) and "theme-group" in body,
            f"/settings serves the settings page (got {status})",
        )

        for asset in ASSETS:
            status, _, _ = asgi_request(app, asset)
            check(status == 200, f"{asset} serves (got {status})")

        # The mount must not swallow API misses: same envelope as ever.
        status, _, body = asgi_request(app, "/api/v1/nope")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"API miss keeps the JSON envelope with static mounted (got {body!r})",
        )


def test_every_page_route_has_a_file_behind_it() -> None:
    """No route may point at an HTML file that is not there.

    This is the check that would have caught the bug this file sat on for
    as long as it existed. ``/run/{id}`` kept its route when the run detail
    page was deleted with the supervised-subprocess route, so ``FileResponse``
    raised at send time and any live server answered a stale bookmark with
    **500** -- while this file, not calling its own tests, reported green.

    Two halves, because either alone is half an answer: the pages that are
    registered must all serve, and the registered set must be exactly the
    HTML files that exist. The second half is the one that notices a *new*
    page added to one side only.
    """
    print("\n== every page route has a file, and every file has a route ==")
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_container(tmp).services, static_dir=FRONTEND)

        # Static checks first, and they gate the requests below. A route
        # whose file is missing makes FileResponse raise *at send time*, so
        # requesting it crashes the file instead of failing a check -- which
        # is how the original 500 stayed invisible. Reported as a failed
        # check, the bug is legible; a traceback out of the test file is
        # not.
        #
        # A route's page is named by its last *non-parameter* segment, so
        # `/monitor/{monitor_id}` and `/datasets/{name}` both resolve to the
        # HTML for the segment before the id.
        page_routes = _page_routes(app)
        orphans = sorted(
            f"{path} -> {name}"
            for path, name in page_routes.items()
            if not (FRONTEND / f"{name}.html").is_file()
        )
        check(
            not orphans,
            f"every page route has an HTML file behind it (orphans: {orphans})",
        )

        on_disk = {p.stem for p in FRONTEND.glob("*.html")}
        unrouted = sorted(on_disk - set(page_routes.values()))
        check(
            not unrouted,
            f"every HTML file is reachable by some route (unrouted: {unrouted})",
        )

        if orphans:
            check(False, "page requests skipped: a route has no file, so "
                         "requesting it raises rather than answering")
            return

        served = ["/", "/graph", "/config", "/datasets", "/help", "/settings",
                  "/monitor/mon-test", "/datasets/anything"]
        for path in served:
            status, _, body = asgi_request(app, path)
            check(
                status == 200 and isinstance(body, str),
                f"{path} serves (got {status})",
            )

        # The retired route answers 404, not 500: a stale bookmark is a
        # missing page, and a 500 would read as a broken server.
        status, _, body = asgi_request(app, "/run/12")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"/run/{{id}} is a plain 404 now its page is gone "
            f"(got {status} {body!r})",
        )


def test_no_static_dir_means_api_only() -> None:
    print("\n== without static_dir, / answers the API envelope ==")
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_container(tmp).services)
        status, _, body = asgi_request(app, "/")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"/ is a 404 envelope when no frontend is wired (got {status} {body!r})",
        )
        status, _, body = asgi_request(app, "/monitor/x")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"monitor page 404s the same way (got {status})",
        )


def test_static_cache_headers() -> None:
    print("\n== pages + assets revalidate (Cache-Control: no-cache) ==")
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(_container(tmp).services, static_dir=FRONTEND)

        for path in ("/", "/graph", "/config", "/datasets",
                     "/help", "/settings", "/monitor/mon-test",
                     "/ui/css/style.css", "/ui/js/editor.js"):
            status, headers, _ = asgi_request(app, path)
            check(
                status == 200 and headers.get("cache-control") == "no-cache",
                f"{path} sends Cache-Control: no-cache "
                f"(got {status}, {headers.get('cache-control')!r})",
            )

        # /api keeps default semantics -- the monitor SSE stream lives there
        # and must not inherit asset caching headers.
        status, headers, _ = asgi_request(app, "/api/v1/health")
        check(
            status == 200 and "cache-control" not in headers,
            f"/api/v1/health has no Cache-Control override (got "
            f"{status}, {headers.get('cache-control')!r})",
        )


def main() -> None:
    """Call every test in this file, then report.

    The file used to end in a bare ``finish()``: the tests above were
    defined and never called, so it ran zero checks and printed
    ``ALL CHECKS PASSED``. Green, on nothing. That is why the 500 below sat
    in the gate unnoticed -- and it is invisible to every gate that only
    looks at exit codes, which is all of them.
    """
    test_pages_and_assets()
    test_no_static_dir_means_api_only()
    test_static_cache_headers()
    test_every_page_route_has_a_file_behind_it()
    finish()


main()
