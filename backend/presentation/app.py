"""Application factory -- builds the FastAPI app from wired services.

``create_app`` receives everything it needs (no imports of
infrastructure, no env reads): the composition root in
``backend.bootstrap`` decides *what* is wired; this module decides
*how* it is exposed over HTTP. Tests call ``create_app`` directly with
fakes and a temp database.
"""

from __future__ import annotations

from fastapi import FastAPI

from .. import __version__
from ..application.services import ApplicationServices
from .api import assets, config, events, health, runs, settings
from .errors import register_error_handlers


def create_app(services: ApplicationServices) -> FastAPI:
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
    app.include_router(events.router)
    return app
