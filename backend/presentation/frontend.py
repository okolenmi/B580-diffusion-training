"""Static frontend serving -- pages + assets, one origin with the API.

Page routes, deliberately *not* a root mount:

* ``/``, ``/monitor/{monitor_id}``, ``/graph``, ``/config``,
  ``/datasets``, ``/datasets/{name}``, ``/help`` and ``/settings`` serve
  the page files (monitor/dataset ids live in the URL; the HTML is
  generic and the client script reads it, like the legacy dashboard);
* ``/run/{run_id}`` used to be here and is deliberately not any more. The
  run detail page went with the supervised-subprocess route (docs 11), and
  the route outlived its ``run.html`` -- so an old bookmark or a stale
  link got a 500 from ``FileResponse`` rather than a 404. Deleting the
  route is the honest answer: there is no run page, and ``test_pages.py``
  now checks that every registered page route has a file behind it, which
  is the check that would have caught the mismatch;
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
        # Exactly the supertype's annotation. Bare `os.PathLike` is
        # `PathLike[Any]`, which is *wider* than the `str | PathLike[str]`
        # Starlette declares, so this was not a valid override (mypy
        # override / Liskov). Narrowing it to match also states that the
        # value is a real path, not an arbitrary one.
        full_path: str | os.PathLike[str],
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

    @app.get("/datasets", include_in_schema=False)
    def datasets_page() -> FileResponse:
        return FileResponse(static_dir / "datasets.html", headers=_NO_CACHE)

    @app.get("/datasets/{name}", include_in_schema=False)
    def dataset_detail_page(name: str) -> FileResponse:
        # One HTML for list + detail; views/datasets.js routes on the path.
        return FileResponse(static_dir / "datasets.html", headers=_NO_CACHE)

    @app.get("/help", include_in_schema=False)
    def help_page() -> FileResponse:
        # Structured placeholder (M8d); no client script beyond the shell.
        return FileResponse(static_dir / "help.html", headers=_NO_CACHE)

    @app.get("/settings", include_in_schema=False)
    def settings_page() -> FileResponse:
        return FileResponse(static_dir / "settings.html", headers=_NO_CACHE)

    @app.get("/setup", include_in_schema=False)
    def setup_page() -> FileResponse:
        """The first-run installer.

        Served unconditionally rather than only when unconfigured: the
        page itself checks `/installer/state` and steps aside with a link to
        Settings if setup is already done. Redirecting here instead would
        make the *page* unreachable after first run, which is a worse
        answer than a page that says so -- and this way the address is
        something a user can always come back to and read.
        """
        return FileResponse(static_dir / "setup.html", headers=_NO_CACHE)

    app.mount("/ui", _UiStaticFiles(directory=static_dir), name="ui")
