#!/usr/bin/env python3
"""Sweep every backend operation for a 5xx.

    python scripts/fuzz_api.py                        # against 127.0.0.1:8766
    python scripts/fuzz_api.py http://127.0.0.1:8799

**Report only, and deliberately not in the gate.** It needs a live server,
it issues ~9k requests, and it runs for minutes -- a gate step has to be
cheap. Run it when the API surface changes.

Why it exists: line coverage and the unit tests both answer "was this code
run", and neither answers "does this operation survive hostile input".
`backend/tests/test_property_boundaries.py` covers three boundaries
(sanitisation, host/origin, path traversal) at the unit level; this asks the
same question at the HTTP boundary, for every operation at once. It found
two real 500s on first contact -- an id past SQLite's INTEGER range, and a
NUL byte reaching `lstat()` -- both of which unit-level tests had no reason
to reach.

A 5xx here is a contract failure rather than a robustness nicety: every
error this API can return is in the table in
`docs/design/backend/02-api-reference.md`, and `internal_error` is the
handler's answer to an exception nobody planned for. The interesting
question is not "does it 500" but "which exception reached the handler",
which the server log names.

Two things it deliberately does not do:

* **Stream endpoints are skipped** -- `/monitor/{id}/stream` and
  `/events`. A successful subscription is an open response by design, so
  "send hostile input, expect a status" has no answer for one: it times
  out, which reads as a failure and is not one. Their error paths are
  `test_api_monitor.py`'s job.
* **It does not assert 4xx codes.** That any specific hostile input maps to
  404 rather than 422 is a per-endpoint decision, and guessing it here
  would produce a wall of false alarms. 4xx is enough: the bug class is an
  exception escaping, not a wrong-but-reasonable status.

**It writes into the repository's real `datasets/`.** There is no override
tier for that directory: `path_tiers.datasets_dir()` is
`project_root / "datasets"` unconditionally, and `project_root` is the
checkout the server runs from. Neither `BACKEND_DB_PATH` nor `COMFY_DIR`
moves it -- `COMFY_DIR` is the *ComfyUI* directory, and it is not even read
by that function. So a sweep that POSTs to `/datasets` creates real dataset
directories. Either run it against a copy of the checkout, or check
`git status` afterwards and delete what appeared.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from urllib.parse import quote

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"

#: Values chosen to reach the boundaries where a parser, a validator or a
#: query builder tends to fall over: embedded NULs, traversal in several
#: encodings, float and integer extremes, format confusion, and control
#: characters that some layers treat as delimiters.
NASTY = [
    "\x00", "%00", "../../etc/passwd", "%2e%2e%2f", "\n", "\r\n",
    "x" * 5000, "-1", "1e400", "NaN", "Infinity",
    "9223372036854775808", "0x10", "1; DROP TABLE graph_executions;--",
    "%zz", "{", "}", '"', "\\", "\x1b[31m", "99999999999999999999",
]

#: Bodies that are wrong in the ways a client can be wrong, rather than
#: merely different: missing keys, right keys of the wrong type, and the
#: shapes a nested schema should reject.
JSON_BODIES = [
    None, {}, [], "string", 12, True, {"a": 1},
    {"name": "\x00"}, {"name": "x" * 100000},
    {"graph": None}, {"graph": {}}, {"graph": []}, {"graph": "x"},
    {"graph": {"format": 1, "nodes": "x", "edges": None}},
    {"graph": {"format": 1, "nodes": [{"id": None}], "edges": []}},
    {"nodes": [{"id": "\x00", "class_name": None}]},
    {"limit": "x"}, {"limit": -1}, {"limit": 1e400}, {"offset": -5},
    {"items": None}, {"items": "x"}, {"items": [{"id": None}]},
    {"path": "../../etc/passwd"}, {"path": "\x00"},
    {"url": "javascript:alert(1)"}, {"url": "file:///etc/passwd"},
    {"kind": "\x00"}, {"name": "../../.."},
]

#: Not JSON at all. A body the framework cannot parse should be a 4xx.
RAW_BODIES = [b"", b"\x00", b"{", b"[1,", b"not json", b"\xff\xfe"]

METHODS_WITH_BODY = {"POST", "PUT", "PATCH"}


def call(method: str, path: str, body=None, raw: bytes | None = None):
    """(status, first bytes of the response). ``None`` status = no response."""
    data = raw if raw is not None else (
        json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status, resp.read()[:200]
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()[:200]
    except Exception as exc:  # noqa: BLE001 -- a non-HTTP failure is a finding
        return None, f"{type(exc).__name__}: {exc}".encode()


def operations() -> list[tuple[str, str]]:
    spec = json.load(urllib.request.urlopen(BASE + "/openapi.json", timeout=15))
    return [
        (method.upper(), path)
        for path, item in spec["paths"].items()
        for method in item
        if method in {"get", "post", "put", "patch", "delete"}
    ]


def probes_for(path: str) -> list[str]:
    """Concrete URLs for one templated path, or the path itself."""
    if "{" not in path:
        return [path]
    var = path[path.index("{") + 1:path.index("}")]
    concrete = path.replace("{" + var + "}", "{v}")
    out = []
    for value in NASTY:
        try:
            encoded = quote(value, safe="")
        except UnicodeEncodeError:
            # A lone surrogate is not encodable, so it cannot be percent
            # encoded either. It still travels in a JSON body, which is
            # where the interesting parsing happens.
            continue
        out.append(concrete.replace("{v}", encoded))
    return out


#: Endpoints whose success is an open response, so "send hostile input and
#: expect a status" has no answer for them -- they would time out rather
#: than answer, which reads as a failure and is not one. Their error paths
#: belong to ``test_api_monitor.py``.
STREAM_PATHS = {"/api/v1/events"}


def is_stream(path: str) -> bool:
    return path.endswith("/stream") or path in STREAM_PATHS


def main() -> int:
    ops = operations()
    print(f"sweeping {len(ops)} operations at {BASE}\n")

    server_errors: list[tuple[str, str, object, int]] = []
    transport: list[tuple[str, str, object, bytes]] = []
    counts: dict[object, int] = {}

    for method, path in ops:
        if is_stream(path):
            print(f"  (skipping {method} {path}: long-lived by design)")
            continue
        for probe in probes_for(path):
            bodies: list[tuple[str, object]] = [
                ("json", b)
                for b in (JSON_BODIES if method in METHODS_WITH_BODY else [None])
            ]
            if method in METHODS_WITH_BODY:
                bodies += [("raw", r) for r in RAW_BODIES]

            for kind, payload in bodies:
                status, response = (
                    call(method, probe, raw=payload) if kind == "raw"
                    else call(method, probe, body=payload))
                counts[status] = counts.get(status, 0) + 1
                if status is None:
                    transport.append((method, probe, payload, response[:120]))
                elif status >= 500:
                    server_errors.append((method, probe, payload, status))

    print("\nstatus distribution:")
    for code in sorted(counts, key=lambda c: (c is None, c)):
        print(f"  {code}: {counts[code]}")

    print(f"\n5xx responses: {len(server_errors)}")
    for (method, shape), probe in sorted(_collapse(server_errors).items()):
        print(f"  {method} {shape}")
        print(f"      e.g. {probe[:100]}")

    print(f"\ntransport-level failures (no HTTP response): {len(transport)}")
    for method, path, body, response in transport[:10]:
        print(f"  {method} {path} body={body!r} -> {response}")

    print(
        "\nA 5xx above is named in the server log with its exception; that\n"
        "traceback is the finding. Re-run with the guard's test removed to\n"
        "confirm a fix, the way the two that were fixed were confirmed."
    )
    return 1 if server_errors or transport else 0


def _collapse(entries):
    """Group by operation and path *shape*, so a long probe value does not
    turn one finding into forty lines."""
    import re

    grouped: dict[tuple[str, str], str] = {}
    for method, path, _body, _status in entries:
        shape = re.sub(r"x{20,}", "<2000 x's>", path)
        grouped.setdefault((method, shape), path)
    return grouped


if __name__ == "__main__":
    sys.exit(main())