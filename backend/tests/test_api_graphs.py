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
services = build_services()
app = create_app(services)

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
check(detail["graph"]["format"] == 2 and len(detail["graph"]["nodes"]) == 2,
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

# -- run: per-execution memory overrides (MEM-02 #2) -------------------
over_body = valid_graph()
over_body["memory_overrides"] = {"vram_max_mb": 4096, "strict": False}
status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST",
                               json_body=over_body)
check(status == 201, "run with memory overrides accepted")
over_id = body["execution_id"]
over_done = wait_until(
    lambda: get_exec(app, over_id)[2]["status"] in TERMINAL, timeout=5.0
)
check(over_done, "override run reaches a terminal state")

# A typo in an override is a 422 naming the key -- it must not fall
# back to the graph's value, because the caller asked to change
# exactly that value.
typo = valid_graph()
typo["memory_overrides"] = {"vram_max": 4096}
status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST", json_body=typo)
check(status == 422, "unknown memory override key refused with 422")
loc = body.get("error", {}).get("details", [{}])[0].get("loc", [])
check(
    "memory_overrides" in loc and "vram_max" in loc,
    f"the 422 points at the offending key (got {loc})",
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
check(body["graph"]["format"] == 2 and body["graph"]["palette_note"] == "keep me",
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

# -- history wipe, on the rows *and* on the disk ---------------------------
# Round-3 N3-07. A row is not the whole of a run: each execution leaves a
# graph.json, an events file and a log in the scratch directory, and
# deleting the history used to remove only the rows. So the directory grew
# by one run's worth forever while the UI reported the history as cleared.
#
# Measured for the sizes this test cares about: one 4000-node execution
# leaves 845,756 bytes (62% events, 38% graph). What matters here is not
# the size but that the files go when the rows do.
scratch_dir = services.graphs.delete_executions.scratch_dir
check(scratch_dir.is_dir(),
      f"the scratch directory the supervisor writes into exists "
      f"({scratch_dir})")

status, _, body = asgi_request(app, f"{GRAPH}/run", method="POST",
                              json_body=valid_graph())
check(status in (200, 201), f"a run to leave scratch behind ({status})")
# The supervisor writes the graph and the event file before the child is
# spawned, so they are there whether or not the run gets anywhere.
started_files = sorted(p.name for p in scratch_dir.iterdir())
check(any(name.endswith(".graph.json") for name in started_files)
      and any(name.endswith(".events.jsonl") for name in started_files),
      f"a run left its graph and event file behind ({started_files})")
on_disk = sum(p.stat().st_size for p in scratch_dir.iterdir()
              if p.is_file())
check(on_disk > 0, f"and they are not empty ({on_disk} bytes)")

status, _, before = asgi_request(app, f"{GRAPH}/executions")
total = before["count"]
status, _, body = asgi_request(app, f"{GRAPH}/executions", method="DELETE")
check(status == 200 and body["deleted"] == total,
      f"delete wipes execution history, including the run just started "
      f"above ({body.get('deleted')}/{total})")
left = sorted(p.name for p in scratch_dir.iterdir())
check(left == [],
      f"and the scratch too, so clearing the history frees the disk "
      f"rather than only the rows (left {left})")
status, _, body = asgi_request(app, f"{GRAPH}/executions")
check(body["count"] == 0, "history empty after wipe")

finish()
