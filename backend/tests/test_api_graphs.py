"""Graph API tests -- end-to-end over raw ASGI (M4).

Two wirings: the real ``build_container`` (palette over the true nodes/
walk -- the bootstrap wiring test) and ``build_services`` (fixture node
classes, so run/stop/library endpoints never touch the GPU).

Run directly: python backend/tests/test_api_graphs.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.bootstrap import build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import (
    asgi_request,
    build_services,
    check,
    concrete_node_classes,
    finish,
    wait_until,
)

TERMINAL = ("finished", "error", "stopped")
GRAPH = "/api/v1/graphs"


def expect_error(status, body, want_status, want_code, label) -> None:
    check(status == want_status, f"{label}: status {want_status} (got {status})")
    code = body.get("error", {}).get("code") if isinstance(body, dict) else None
    check(code == want_code, f"{label}: envelope code {want_code!r} (got {code!r})")


def valid_graph() -> dict:
    return {
        "nodes": [
            {"id": "v", "class_name": "ScaleNode",
             "params": {"value": 3.0, "factor": 5.0}},
            {"id": "s", "class_name": "SumNode", "params": {"b": 1.0}},
        ],
        "edges": [
            {"from_node": "v", "from_port": "scaled",
             "to_node": "s", "to_port": "a"},
        ],
    }


def get_exec(app, execution_id):
    return asgi_request(app, f"{GRAPH}/executions/{execution_id}")


# ==========================================================================
# Section A: real composition root -- palette over the real scan
# ==========================================================================
root_a = Path(tempfile.mkdtemp(prefix="backend-api-graph-a-"))
container = build_container(
    Settings(project_root=root_a, db_path=root_a / "backend.db")
)
app_a = create_app(container.services)

status, _, body = asgi_request(app_a, f"{GRAPH}/nodes")
check(status == 200, "real container serves the palette")
check(
    body["count"] == len(concrete_node_classes()),
    f"palette serves every concrete Node subclass (got {body['count']})",
)
check(body["load_errors"] == [], "no discovery failures reported")
check("optimizer" in body["domains"] and "dataset" in body["domains"],
      "domains grouped from module paths")

status, _, body = asgi_request(app_a, f"{GRAPH}/executions")
check(status == 200 and body == {"executions": [], "count": 0},
      "startup reconcile left no unfinished rows; history starts empty")

# ==========================================================================
# Section B: fixture services -- the full endpoint matrix
# ==========================================================================
app = create_app(build_services())

# -- catalog ---------------------------------------------------------------
status, _, body = asgi_request(app, f"{GRAPH}/nodes")
check(status == 200 and body["load_errors"] == [], "fixture palette serves clean")
by_name = {n["class_name"]: n for ns in body["domains"].values() for n in ns}
sum_node = by_name["SumNode"]
check(
    sum_node["display_name"] == "Sum"
    and sum_node["inputs"][0]["required"] is True
    and sum_node["inputs"][0]["default"] is None,
    "required input: no default on the wire",
)
scale = {p["name"]: p for p in by_name["ScaleNode"]["inputs"]}
check(
    scale["factor"]["default"] == 2.0 and scale["factor"]["default_repr"] == "2.0",
    "optional default: JSON value + repr together",
)
check(scale["mode"]["choices"] == ["mul", "div"], "choices exposed as a list")
check(
    by_name["PresetChoiceNode"]["presets"][0]["name"] == "identity",
    "dynamic node carries presets",
)
check(by_name["SumNode"]["presets"] is None, "static node: presets null")

# -- diagnostics -----------------------------------------------------------
status, _, body = asgi_request(
    app, f"{GRAPH}/nodes/DiagnosingNode/diagnostics",
    method="POST", json_body={"params": {"path": "ckpt.pt"}},
)
check(status == 200 and body["messages"] == {"path": ["looked at 'ckpt.pt'"]},
      "diagnostics returns its lines")
status, _, body = asgi_request(
    app, f"{GRAPH}/nodes/BadDiagnosticsNode/diagnostics", method="POST", json_body={}
)
expect_error(status, body, 400, "node_diagnostics_failed",
             "raising diagnostics() is 400, not 500")
status, _, body = asgi_request(
    app, f"{GRAPH}/nodes/NoSuchNode/diagnostics", method="POST", json_body={}
)
expect_error(status, body, 404, "node_class_not_found", "unknown class diagnostics")

# -- validate --------------------------------------------------------------
status, _, body = asgi_request(
    app, f"{GRAPH}/validate", method="POST", json_body=valid_graph()
)
check(status == 200 and body["ok"] is True and body["issues"] == [],
      "valid graph: 200 ok with no issues")

bad_graph = {"nodes": [{"id": "s", "class_name": "NoSuchNode"}], "edges": []}
status, _, body = asgi_request(app, f"{GRAPH}/validate", method="POST",
                               json_body=bad_graph)
check(status == 200 and body["ok"] is False,
      "invalid graph still answers 200 on validate (ok=false)")
check(
    [i["code"] for i in body["issues"]] == ["unknown_class"],
    f"unknown class localized (got {[i['code'] for i in body['issues']]})",
)
# Params of an unknown class can't be checked -- its shape is unknown.
status, _, body = asgi_request(
    app, f"{GRAPH}/validate",
    method="POST",
    json_body={"nodes": [{"id": "s", "class_name": "SumNode"}], "edges": []},
)
check(
    [i["code"] for i in body["issues"]]
    == ["missing_required_input", "missing_required_input"],
    "known class with no providers: one issue per required input",
)

# -- run: rejection --------------------------------------------------------
status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST",
                               json_body=bad_graph)
expect_error(status, body, 422, "graph_invalid", "invalid run refused")
details = body.get("error", {}).get("details")
check(
    isinstance(details, list) and details
    and {"severity", "code", "message"} <= set(details[0]),
    "graph_invalid details carry every issue",
)

# -- run: happy path -------------------------------------------------------
status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST",
                               json_body=valid_graph())
check(status == 201 and body["status"] in ("queued", "running"),
      f"valid run answers 201 queued (got {status}/{body.get('status')})")
first_id = body["execution_id"]

finished = wait_until(
    lambda: get_exec(app, first_id)[2]["status"] in TERMINAL, timeout=5.0
)
check(finished, "execution reaches a terminal state")
status, _, detail = get_exec(app, first_id)
check(detail["status"] == "finished", "clean graph ends finished")
check(
    [r["node_id"] for r in detail["results"]] == ["v", "s"]
    and all(r["ok"] for r in detail["results"]),
    "full detail carries per-node results",
)
check(detail["graph"]["format"] == 1 and len(detail["graph"]["nodes"]) == 2,
      "full detail carries the submission snapshot")
check(
    all(isinstance(r["duration_ms"], (int, float)) for r in detail["results"]),
    "durations are numbers",
)

status, _, listing = asgi_request(app, f"{GRAPH}/executions")
check(listing["count"] == 1, "history lists the execution")
check(
    "results" not in listing["executions"][0]
    and "graph" not in listing["executions"][0],
    "list rows are summaries (no results/graph payload)",
)
status, _, body = asgi_request(app, f"{GRAPH}/executions?limit=0")
expect_error(status, body, 422, "invalid_query", "limit out of bounds")
status, _, body = asgi_request(app, f"{GRAPH}/executions/999999")
expect_error(status, body, 404, "graph_execution_not_found", "unknown execution")

# An id a URL can carry but an INTEGER column cannot hold. Found by
# fuzzing every operation for 5xx: SQLite's INTEGER is signed 64-bit and
# the driver raises OverflowError rather than truncating, so binding
# 2**63 reached the client as a 500. No row can have that id, so it is
# "not found" -- the same answer as 999999 above, for a value that is
# merely further out of range rather than merely unused.
for too_big, label in (
    (2 ** 63, "one past the signed 64-bit maximum"),
    (2 ** 64, "one past the unsigned 64-bit maximum"),
    (10 ** 30, "absurdly large"),
):
    status, _, body = asgi_request(app, f"{GRAPH}/executions/{too_big}")
    expect_error(status, body, 404, "graph_execution_not_found",
                 f"execution id past SQLite's range: {label}")
    status, _, body = asgi_request(app, f"{GRAPH}/executions/{too_big}/stop",
                                   method="POST")
    expect_error(status, body, 404, "graph_execution_not_found",
                 f"stopping an execution id past SQLite's range: {label}")

# The boundary itself must still work: 2**63 - 1 is representable.
status, _, body = asgi_request(app, f"{GRAPH}/executions/{2 ** 63 - 1}")
expect_error(
    status, body, 404, "graph_execution_not_found",
    "the largest representable id is a normal lookup, not a range error",
)

# -- single-active + stop --------------------------------------------------
status, _, body = asgi_request(
    app, f"{GRAPH}/run",
    method="POST",
    json_body={"nodes": [{"id": "slow", "class_name": "SlowNode",
                          "params": {"seconds": 0.4}}],
               "edges": []},
)
check(status == 201, "slow run accepted")
slow_id = body["execution_id"]

status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST",
                               json_body=valid_graph())
expect_error(status, body, 409, "graph_execution_active",
             "second run refused while one is active")

status, _, body = asgi_request(app, f"{GRAPH}/executions/{slow_id}/stop",
                               method="POST")
check(status == 200 and body["status"] == "stopped"
      and body["error"] == "stop requested",
      "stop answers 200 with the stopped row")
status, _, body = asgi_request(app, f"{GRAPH}/executions/{slow_id}/stop",
                               method="POST")
expect_error(status, body, 409, "graph_execution_not_active",
             "second stop is 409 with the winner status")
status, _, body = asgi_request(app, f"{GRAPH}/executions/999999/stop",
                               method="POST")
expect_error(status, body, 404, "graph_execution_not_found",
             "stopping unknown execution")

import time

time.sleep(0.5)  # let the stopped worker exit before history is wiped

# -- library ---------------------------------------------------------------
status, _, body = asgi_request(app, f"{GRAPH}/library")
check(status == 200 and body == {"graphs": [], "count": 0},
      "library starts empty")

stored = dict(valid_graph())
stored["palette_note"] = "keep me"
status, _, body = asgi_request(
    app, f"{GRAPH}/library/{quote(' my graph ')}",
    method="PUT", json_body=stored,
)
check(status == 201, "first save answers 201")
check(body["name"] == "my graph" and body["node_count"] == 2,
      "name trimmed, node_count derived")
check(body["graph"]["format"] == 1 and body["graph"]["palette_note"] == "keep me",
      "format stamped; unknown keys preserved verbatim")

status, _, body = asgi_request(
    app, f"{GRAPH}/library/{quote(' my graph ')}",
    method="PUT", json_body=stored,
)
check(status == 200, "replace answers 200")

status, _, body = asgi_request(app, f"{GRAPH}/library")
check(body["count"] == 1 and "graph" not in body["graphs"][0],
      "library list rows are summaries")

# Saving never validates: an unknown class still stores, then fails at run.
future = {"nodes": [{"id": "n", "class_name": "ClassNotWrittenYet", "params": {}}],
          "edges": []}
status, _, body = asgi_request(
    app, f"{GRAPH}/library/future", method="PUT", json_body=future
)
check(status == 201, "graph referencing a missing class saves fine")
status, _, body = asgi_request(app, f"{GRAPH}/library/future")
saved_payload = body["graph"]
status, _, body = asgi_request(
    app, f"{GRAPH}/run", method="POST", json_body=saved_payload
)
expect_error(status, body, 422, "graph_invalid",
             "the same graph fails validation at run time")

status, _, body = asgi_request(app, f"{GRAPH}/library/missing")
expect_error(status, body, 404, "graph_not_found", "unknown saved graph")
status, _, body = asgi_request(app, f"{GRAPH}/library/{quote(' ')}",
                               method="PUT", json_body=stored)
expect_error(status, body, 422, "invalid_query", "blank name refused")

status, _, body = asgi_request(app, f"{GRAPH}/library/future", method="DELETE")
check(status == 200 and body["deleted"] is True, "delete removes the graph")
status, _, body = asgi_request(app, f"{GRAPH}/library/future", method="DELETE")
expect_error(status, body, 404, "graph_not_found", "deleting twice is 404")

# -- history wipe ----------------------------------------------------------
status, _, body = asgi_request(app, f"{GRAPH}/executions", method="DELETE")
check(status == 200 and body["deleted"] == 2,
      f"delete wipes execution history ({body.get('deleted')})")
status, _, body = asgi_request(app, f"{GRAPH}/executions")
check(body["count"] == 0, "history empty after wipe")

finish()
