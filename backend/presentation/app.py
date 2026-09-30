"""Application factory -- builds the FastAPI app from wired services.

``create_app`` receives everything it needs (no imports of
infrastructure, no env reads): the composition root in
``backend.bootstrap`` decides *what* is wired; this module decides
*how* it is exposed over HTTP. Tests call ``create_app`` directly with
fakes and a temp database.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from .. import __version__
from ..application.services import ApplicationServices
from .api import (
    assets,
    config,
    datasets,
    events,
    graphs,
    health,
    monitor,
    runs,
    settings,
)
from .errors import register_error_handlers
from .frontend import register_frontend


def create_app(
    services: ApplicationServices, *, static_dir: Path | None = None
) -> FastAPI:
    app = FastAPI(
        title="Training Backend",
        version=__version__,
        description="Clean-room replacement for the legacy training server.",
    )
    app.state.services = services

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(runs.router)
    app.include_router(config.router)
    app.include_router(settings.router)
    app.include_router(assets.router)
    app.include_router(datasets.router)
    app.include_router(graphs.router)
    app.include_router(events.router)
    app.include_router(monitor.router)
    # Page/asset routes last: they never shadow the API, and API misses
    # keep their JSON error envelope (no static 404 HTML under /api/).
    if static_dir is not None:
        register_frontend(app, static_dir)
    return app
