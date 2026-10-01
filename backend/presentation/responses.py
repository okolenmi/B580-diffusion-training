"""Presentation response classes.

``SanitizingJSONResponse`` is the app-wide REST default: every JSON body
passes through :mod:`backend.json_safe`, so a non-finite float that
reached the database (a diverged trainer's ``loss``, docs 07 F-03)
becomes ``null`` plus a ``nonfinite`` marker instead of
``allow_nan=False``'s ``ValueError`` -- which used to 500 the whole page.
"""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse

from ..json_safe import sanitize, strict_dumps


class SanitizingJSONResponse(JSONResponse):
    """``JSONResponse`` that never emits non-JSON constants."""

    def render(self, content: Any) -> bytes:
        return strict_dumps(sanitize(content)).encode("utf-8")