"""Static frontend serving -- pages + assets, one origin with the API.

Three registrations, deliberately *not* a root mount:

* ``/`` and ``/monitor/{monitor_id}`` serve the two page files (the
  monitor page reads its id from the URL, like the legacy dashboard);
* ``/ui/*`` serves the frontend directory (ES modules, css);
* nothing catches ``/api/...`` misses, so unknown API routes keep
  answering with the JSON error envelope instead of static 404 HTML.

Tests never pass ``static_dir``; the CLI does.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)


def register_frontend(app: FastAPI, static_dir: Path) -> None:
    """Attach page routes and the ``/ui`` asset mount if the directory exists."""
    if not static_dir.is_dir():
        logger.warning(
            "frontend directory %s missing -- API-only (pages will 404)", static_dir
        )
        return

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/monitor/{monitor_id}", include_in_schema=False)
    def monitor_page(monitor_id: str) -> FileResponse:
        # The id is the page's own URL segment; the HTML is generic and
        # the client script reads it (mirrors server/main.py's route).
        return FileResponse(static_dir / "monitor.html")

    @app.get("/graph", include_in_schema=False)
    def graph_page() -> FileResponse:
        return FileResponse(static_dir / "graph.html")

    app.mount("/ui", StaticFiles(directory=static_dir), name="ui")
