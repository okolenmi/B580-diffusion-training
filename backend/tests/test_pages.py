"""Frontend serving tests -- page routes + /ui asset mount (M6/M7).

Pins register_frontend's contract: pages serve when ``static_dir`` is
given, every asset the shipped pages reference resolves, and the mount
can never shadow the API's error envelope (02-api-reference.md section 1).

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
    "/ui/css/monitor.css",
    "/ui/css/editor.css",
    "/ui/js/api.js",
    "/ui/js/monitor.js",
    "/ui/js/lib/loss_chart.js",
    "/ui/js/views/dashboard.js",
    "/ui/js/editor.js",
    "/ui/js/editor/state.js",
    "/ui/js/editor/canvas.js",
    "/ui/js/editor/inspector.js",
    "/ui/js/editor/palette.js",
    "/ui/js/editor/executions.js",
    "/ui/js/editor/library.js",
)


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

        for asset in ASSETS:
            status, _, _ = asgi_request(app, asset)
            check(status == 200, f"{asset} serves (got {status})")

        # The mount must not swallow API misses: same envelope as ever.
        status, _, body = asgi_request(app, "/api/v1/nope")
        check(
            status == 404 and isinstance(body, dict) and "error" in body,
            f"API miss keeps the JSON envelope with static mounted (got {body!r})",
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


finish()
