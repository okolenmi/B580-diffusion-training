"""Request guard -- Host header and Origin checks (docs 07 F-06).

The API has no authentication, and that is a documented decision for a
single-user tool bound to loopback. Two browser-level holes had to be
closed anyway, because neither needs credentials:

**DNS rebinding.** A page on ``evil.example`` resolves its own name to
``127.0.0.1`` and then talks to this server with a *valid* session from
the browser's point of view. The ``Host`` header is what tells the two
apart: a request that says ``evil.example`` is not ours, whatever
address it connected to. Loopback names (plus anything the operator
listed in ``BACKEND_ALLOWED_HOSTS``) are the only ones accepted.

**Cross-site writes.** A page on any origin can POST to
``127.0.0.1:8766`` with a simple content type, and the browser sends
it. State-changing methods therefore carry an ``Origin`` check: when the
header is present it must match this server's own origin (or an entry
in ``BACKEND_ALLOWED_ORIGINS``). A request *without* an Origin header is
not browser-initiated cross-site (curl, scripts and the app's own
same-origin fetches are fine) and passes.

Both lists are configurable because the API is documented as reachable
from another machine on the LAN:

* ``BACKEND_ALLOWED_HOSTS`` -- comma-separated extra Host names/addresses.
* ``BACKEND_ALLOWED_ORIGINS`` -- comma-separated extra origins
  (``scheme://host[:port]``).

Refusals answer 403 with the ordinary error envelope -- never a CORS
preflight, never a redirect.

Implemented as raw ASGI (not ``@app.middleware("http")``) on purpose:
FastAPI's helper wraps the ``receive`` channel to cache the body, which
consumes ``http.disconnect`` and deadlocks the SSE endpoints' liveness
check. This class only reads headers and otherwise passes the scope,
receive and send through untouched.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .errors import error_body

logger = logging.getLogger(__name__)

DEFAULT_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")

# Methods that change state. GET/HEAD stay open: a hostile page can only
# read with those, and reads are scoped to the loopback deployment.
_STATE_CHANGING = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _split_env(name: str) -> tuple[str, ...]:
    raw = os.environ.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def host_name(value: str) -> str:
    """Hostname part of a ``Host`` header: strips the port, keeps IPv6.

    ``127.0.0.1:8766`` and ``127.0.0.1`` name the same server, and the
    port carries no meaning for the rebinding defence (it changes where
    the browser connects, not who is asking).
    """
    value = value.strip().lower()
    if value.startswith("["):  # IPv6 literal: [::1] or [::1]:8766
        end = value.find("]")
        return value[: end + 1] if end != -1 else value
    if value.count(":") == 1:
        return value.split(":", 1)[0]
    return value


def allowed_hosts() -> frozenset[str]:
    return frozenset(DEFAULT_HOSTS) | frozenset(
        host_name(host) for host in _split_env("BACKEND_ALLOWED_HOSTS")
    )


def allowed_origins() -> frozenset[str]:
    return frozenset(
        origin.lower().rstrip("/") for origin in _split_env("BACKEND_ALLOWED_ORIGINS")
    )


class HostOriginGuard:
    """Pure pass-through ASGI middleware: inspect, refuse, or forward."""

    def __init__(
        self,
        app: ASGIApp,
        hosts: frozenset[str],
        origins: frozenset[str],
    ) -> None:
        self._app = app
        self._hosts = hosts
        self._origins = origins

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = headers.get("host", "").strip().lower()
        if host and host_name(host) not in self._hosts:
            logger.warning("refused request with Host %r", host)
            await self._refuse(
                scope, receive, send,
                "forbidden_host",
                f"Host {host!r} is not served by this backend; expected one "
                f"of {', '.join(sorted(self._hosts))}",
            )
            return

        if scope.get("method", "GET") in _STATE_CHANGING:
            origin = headers.get("origin", "").strip().lower().rstrip("/")
            if origin and origin not in self._origins and not self._is_self(
                origin, host
            ):
                logger.warning(
                    "refused %s from Origin %r", scope.get("method"), origin
                )
                await self._refuse(
                    scope, receive, send,
                    "forbidden_origin",
                    f"Origin {origin!r} may not change state on this backend",
                )
                return

        await self._app(scope, receive, send)

    @staticmethod
    def _is_self(origin: str, host: str) -> bool:
        return origin in (f"http://{host}", f"https://{host}")

    @staticmethod
    async def _refuse(
        scope: Scope, receive: Receive, send: Send, code: str, message: str
    ) -> None:
        await JSONResponse(
            status_code=403, content=error_body(code, message)
        )(scope, receive, send)


def register_request_guard(app: FastAPI) -> None:
    """Install the Host/Origin guard on ``app`` (call before serving)."""
    hosts = allowed_hosts()
    origins = allowed_origins()
    logger.info(
        "request guard: hosts=%s origins=%s",
        ", ".join(sorted(hosts)),
        ", ".join(sorted(origins)) or "(same-origin only)",
    )
    app.add_middleware(HostOriginGuard, hosts=hosts, origins=origins)