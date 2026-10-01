"""json_safe -- the one place a payload is made serializable.

Non-finite floats (``NaN``, ``inf``, ``-inf``) are valid Python, valid
input to ``json.loads`` and valid TOML -- but **not valid JSON**.
``json.dumps`` happily emits the bare token ``NaN``, and every strict
parser rejects it: the browser's ``JSON.parse``, ``jq``, and the
strict parser used in the tests. A diverged trainer writes exactly
that into its progress file (``{"loss": NaN}`` parses fine on the way
in), and from there it lands in the run row, the SSE stream and the
monitor frames (docs 07 F-03).

Two failure modes existed before this module, both bad for a
monitoring tool: the frame carried invalid bytes, so the browser threw
and the frontend handlers swallowed it -- the diverged step simply
vanished while the run still looked alive; and Starlette's JSON
response uses ``allow_nan=False``, so a REST read of the same row
raised ``ValueError`` and turned the whole page into a 500.

Policy: a non-finite float becomes ``null`` (no measurement was
taken), and every object that carried one says so locally -- a
sibling ``nonfinite`` map of ``{key: "nan" | "inf" | "-inf"}`` -- so
the UI can render a loud "diverged" state instead of an em dash that
would claim nothing was ever measured. Marking each level (rather
than one root-level path map) is what makes ``{"runs": [...]}``
usable: a list item names its own ``current_loss``. A non-finite
float inside a bare array has no object to hang the marker on and is
simply nulled.

Dumping runs with ``allow_nan=False``: if the walk ever missed a
value, that raises here instead of shipping a frame the browser
cannot parse.

Used at the three serialization boundaries:
``presentation.responses.SanitizingJSONResponse`` (app-wide REST
default), ``presentation.sse.serialize_event`` (domain events) and
``infrastructure.monitor_bus.SharedMonitorBus.report`` (node
telemetry frames).

Pure functions, no I/O, no layer imports -- importable from every
layer, like :mod:`backend.config`.
"""

from __future__ import annotations

import json
import math
from typing import Any

# What a non-finite float became, per key of the object that held it.
NonFiniteMap = dict[str, str]


def _kind(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _is_nonfinite(value: Any) -> bool:
    # bool is not a float here: `isinstance(True, float)` is False and
    # a JSON true/false is a perfectly good measurement flag.
    return isinstance(value, float) and not math.isfinite(value)


def sanitize(value: Any) -> Any:
    """Deep-copy ``value`` with non-finite floats replaced by ``None``.

    Every dict that held one gains a ``nonfinite`` key mapping the
    local field names to ``"nan" | "inf" | "-inf"``. Values of unknown
    types pass through untouched -- :func:`strict_dumps` is the last
    word.
    """
    if _is_nonfinite(value):
        return None
    if isinstance(value, dict):
        clean: dict[Any, Any] = {}
        hits: NonFiniteMap = {}
        for key, item in value.items():
            if _is_nonfinite(item):
                clean[key] = None
                hits[str(key)] = _kind(item)
            else:
                clean[key] = sanitize(item)
        if hits:
            clean["nonfinite"] = hits
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def strict_dumps(payload: Any) -> str:
    """``json.dumps`` that refuses to emit ``NaN``/``inf`` tokens.

    Feed it :func:`sanitize` output. The explicit ``allow_nan=False``
    is the invariant guard: if the walk ever misses a value, this
    raises here instead of shipping a frame the browser cannot parse.
    """
    return json.dumps(payload, allow_nan=False)