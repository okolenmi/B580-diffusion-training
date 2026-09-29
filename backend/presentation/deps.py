"""FastAPI dependencies -- how handlers reach the wired use cases."""

from __future__ import annotations

from fastapi import Request

from ..application.services import ApplicationServices


def get_services(request: Request) -> ApplicationServices:
    """Return the services aggregate stored by ``create_app``.

    The one sanctioned way for a handler to obtain collaborators --
    handlers never construct repositories, buses, or connections.
    """
    return request.app.state.services
