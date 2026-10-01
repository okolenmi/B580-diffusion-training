"""Static frontend serving -- pages + assets, one origin with the API.

Four registrations, deliberately *not* a root mount:

* ``/``, ``/monitor/{monitor_id}``, ``/graph`` and ``/config`` serve
  the page files (the monitor page reads its id from the URL, like the
  legacy dashboard);
* ``/ui/*`` serves the frontend directory (ES modules, css);
* nothing catches ``/api/...`` misses, so unknown API routes keep
  answering with the JSON error envelope instead of static 404 HTML.

Every response here carries ``Cache-Control: no-cache`` -- ported from
``server/main.py``, which documented why: without it the browser applies
heuristic freshness (age/10 from Last-Modified), so a plain reload can
serve yesterday's JS against today's server -- a real mixed-version bug
hit once in the legacy app (old ``nodegraph.js`` vs new gated widgets).
no-cache still caches; it forces a cheap 304 when the file is unchanged.
``/api`` responses are untouched (SSE included).

Tests never pass ``static_dir``; the CLI does.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

logger = logging.getLogger(__name__)

# Revalidate-always marker; see the module docstring.
_NO_CACHE = {"Cache-Control": "no-cache"}


class _UiStaticFiles(StaticFiles):
    """``StaticFiles`` that stamps ``no-cache`` on every response.

    A subclass because this Starlette build's ``StaticFiles`` takes no
    ``headers`` kwarg; ``file_response`` is the single path both the 200
    and the 304 (``NotModifiedResponse``) go through, so revalidation
    stays cheap and correct for both.
    """

    def file_response(
        self,
        full_path: os.PathLike,
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        response.headers["Cache-Control"] = "no-cache"
        return response


def register_frontend(app: FastAPI, static_dir: Path) -> None:
    """Attach page routes and the ``/ui`` asset mount if the directory exists."""
    if not static_dir.is_dir():
        logger.warning(
            "frontend directory %s missing -- API-only (pages will 404)", static_dir
        )
        return

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html", headers=_NO_CACHE)

    @app.get("/monitor/{monitor_id}", include_in_schema=False)
    def monitor_page(monitor_id: str) -> FileResponse:
        # The id is the page's own URL segment; the HTML is generic and
        # the client script reads it (mirrors server/main.py's route).
        return FileResponse(static_dir / "monitor.html", headers=_NO_CACHE)

    @app.get("/graph", include_in_schema=False)
    def graph_page() -> FileResponse:
        return FileResponse(static_dir / "graph.html", headers=_NO_CACHE)

    @app.get("/config", include_in_schema=False)
    def config_page() -> FileResponse:
        return FileResponse(static_dir / "config.html", headers=_NO_CACHE)

    app.mount("/ui", _UiStaticFiles(directory=static_dir), name="ui")
