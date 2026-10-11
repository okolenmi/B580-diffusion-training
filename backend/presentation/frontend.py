"""Static frontend serving -- pages + assets, one origin with the API.

Page routes, deliberately *not* a root mount:

* ``/``, ``/monitor/{monitor_id}``, ``/graph``, ``/templates[/{name}]``, ``/config``,
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
``archive/server/main.py``, which documented why: without it the browser applies
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
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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


# (route, html file) for the plain static pages. Path parameters are fine
# (`/templates/{name}`) -- the handler ignores them and serves the one file.
_STATIC_PAGES: tuple[tuple[str, str], ...] = (
    ("/graph", "graph.html"),
    ("/templates", "templates.html"),
    ("/templates/{name}", "templates.html"),
    ("/config", "config.html"),
    ("/datasets", "datasets.html"),
    ("/datasets/{name}", "datasets.html"),
    ("/help", "help.html"),
    ("/settings", "settings.html"),
    # The first-run installer. Served unconditionally rather than only when
    # unconfigured: the page itself checks `/installer/state` and steps aside
    # with a link to Settings if setup is already done. Redirecting here
    # instead would make the *page* unreachable after first run, which is a
    # worse answer than a page that says so -- and this way the address is
    # something a user can always come back to and read.
    ("/setup", "setup.html"),
)


def _register_static_page(app: FastAPI, static_dir: Path, route: str, filename: str):
    """Register one `route -> filename.html` page handler on `app`.

    The handler declares the route's own path parameters by name, because
    FastAPI resolves those from the signature: a `**kwargs` catch-all would
    instead read them as query parameters and 422 every `/templates/{name}`.
    Parameterless routes get a no-argument handler.
    """
    params = re.findall(r"{(\w+)}", route)
    response = lambda: FileResponse(static_dir / filename, headers=_NO_CACHE)  # noqa: E731

    if params:
        # Build the signature FastAPI introspects: one `name: str` per path
        # parameter, in the order they appear in the route.
        namespace: dict[str, Any] = {"__annotations__": {p: str for p in params}}
        exec(  # noqa: S102 -- fixed shape: `p0, p1: str` names from `params`
            f"def handler({', '.join(params)}):\n"
            "    return _response()\n",
            {"_response": response},
            namespace,
        )
        handler = cast("Callable[..., FileResponse]", namespace["handler"])
    else:
        def handler() -> FileResponse:
            return response()

    handler.__name__ = (
        "static_page_" + (re.sub(r"\W+", "_", route).strip("_") or "index")
    )
    app.get(route, include_in_schema=False)(handler)


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

    # The remaining pages are one static HTML each, so they are registered
    # from a table instead of ten near-identical functions. Order is the
    # table's order (route registration order is the OpenAPI listing
    # order). Several paths deliberately share one file: the client script
    # routes on the path (same pattern as /datasets/{name}).
    for route, filename in _STATIC_PAGES:
        _register_static_page(app, static_dir, route, filename)

    app.mount("/ui", _UiStaticFiles(directory=static_dir), name="ui")
