"""Dataset API tests -- end-to-end over raw ASGI.

Two wirings: the real ``build_container`` (read paths + error envelopes
+ the catalog-only assets kind) and ``build_services`` (the fake fork
gateway, so task lifecycle endpoints never spawn a child).

Run directly: python backend/tests/test_api_datasets.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.bootstrap import build_container
from backend.config import Settings
from backend.presentation.app import create_app
from backend.tests.support import (
    asgi_request,
    build_services,
    check,
    finish,
    make_v1_dataset,
    make_v2_dataset,
)


def expect_error(status, body, want_status, want_code, label) -> None:
    check(status == want_status, f"{label}: status {want_status} (got {status})")
    code = body.get("error", {}).get("code") if isinstance(body, dict) else None
    check(code == want_code, f"{label}: envelope code {want_code!r} (got {code!r})")


# ==========================================================================
# Section A: real composition root (reads, envelopes, assets kind)
# ==========================================================================

root_a = Path(tempfile.mkdtemp(prefix="backend-api-ds-a-"))
container = build_container(Settings(project_root=root_a, db_path=root_a / "backend.db"))
app = create_app(container.services)

status, _, body = asgi_request(app, "/api/v1/datasets")
check(status == 200 and body == {"datasets": [], "count": 0},
      "empty library lists nothing through the API")

make_v2_dataset(root_a, "api-ds", items=2)
status, _, body = asgi_request(app, "/api/v1/datasets")
check(status == 200 and body["count"] == 1, "fixture dataset listed")
entry = body["datasets"][0]
check(entry["info"]["format_version"] == 2, "summary reports format_version 2")
check(entry["stats"]["items"] == 2 and entry["stats"]["pending"] == 2,
      "summary carries real stats")

status, _, body = asgi_request(app, "/api/v1/datasets/missing")
expect_error(status, body, 404, "dataset_not_found", "GET unknown dataset")

make_v1_dataset(root_a, "old-ds")
status, _, body = asgi_request(app, "/api/v1/datasets/old-ds")
expect_error(status, body, 409, "dataset_not_migrated", "GET v1 dataset")

status, _, body = asgi_request(app, "/api/v1/datasets")
by_name = {d["info"]["name"]: d for d in body["datasets"]}
check(by_name["old-ds"]["stats"] is None and by_name["old-ds"]["info"]["format_version"] == 0,
      "v1 listed with null stats, never fabricated counts")

# preview file serving (M8c): scoped bytes, envelope on refusal
(root_a / "escape.txt").write_text("SECRET")
status, headers, body = asgi_request(
    app, "/api/v1/datasets/api-ds/files/previews/p1.png"
)
check(status == 200, f"preview file serves (got {status})")
check(headers.get("content-type", "").startswith("image/png"),
      f"preview content-type image/png (got {headers.get('content-type')!r})")
check(isinstance(body, str) and "PNG" in body, "PNG signature present in bytes")
check(headers.get("cache-control") == "no-cache", "preview revalidates (no-cache)")

status, _, body = asgi_request(app, "/api/v1/datasets/api-ds/files/previews/none.png")
expect_error(status, body, 404, "dataset_file_not_found", "GET missing preview")

status, _, body = asgi_request(app, "/api/v1/datasets/nope/files/previews/p1.png")
expect_error(status, body, 404, "dataset_not_found", "GET preview, unknown dataset")

status, _, body = asgi_request(app, "/api/v1/datasets/api-ds/files/../../escape.txt")
expect_error(status, body, 404, "dataset_file_not_found", "traversal refused as not-found")
check("SECRET" not in str(body), "escape file never read")

# assets: catalog-only dataset kind
status, _, body = asgi_request(app, "/api/v1/assets/dataset")
check(status == 200, "asset catalog serves kind 'dataset'")
names = {option["value"] for option in body["options"]}
check(names == {"api-ds", "old-ds"}, "catalog lists both dataset names")
check(body["upload_supported"] is False and body["browse_supported"] is False,
      "dataset catalog is read-only")

status, _, body = asgi_request(app, "/api/v1/assets/dataset/browse")
expect_error(status, body, 422, "invalid_query", "browse of dataset kind refused")
status, _, body = asgi_request(
    app, "/api/v1/assets/dataset/folders/x", method="PUT"
)
expect_error(status, body, 422, "invalid_query", "folder creation refused")

# ==========================================================================
# Section B: composition-root twin with the fake fork gateway
# ==========================================================================

root_b = Path(tempfile.mkdtemp(prefix="backend-api-ds-b-"))
services = build_services(project_root=root_b)
app = create_app(services)

status, _, body = asgi_request(
    app, "/api/v1/datasets",
    method="POST", json_body={"name": "made", "description": "via api"},
)
check(status == 201 and body["name"] == "made" and body["format_version"] == 2,
      "POST create returns the new identity (201)")

status, _, body = asgi_request(
    app, "/api/v1/datasets", method="POST", json_body={"name": "made"}
)
expect_error(status, body, 409, "dataset_exists", "duplicate create")

status, _, body = asgi_request(app, "/api/v1/datasets/made")
check(status == 200 and body["info"]["description"] == "via api"
      and body["stats"]["items"] == 0 and body["sets"] == [] and body["active_tasks"] == [],
      "GET detail round-trip (identity + zeroed stats + empty sets/tasks)")

make_v2_dataset(root_b, "flow", items=4)

# -- items -----------------------------------------------------------------

status, _, body = asgi_request(app, "/api/v1/datasets/flow/items")
check(status == 200 and body["count"] == 4, "list items")
check(body["items"][0]["prompt"] == "photo 1" and body["items"][0]["neg_prompt"] == "",
      "item columns on the wire")

status, _, body = asgi_request(app, "/api/v1/datasets/flow/items?committed=false")
check(body["count"] == 4, "membership filter (pending)")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items/1", method="PATCH", json_body={"type": "bad"}
)
check(status == 200 and body["type"] == "bad", "PATCH single flips type")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items/1", method="PATCH", json_body={"prompt": ""}
)
check(status == 200 and body["prompt"] == "", "PATCH single clears prompt with ''")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items/2", method="PATCH", json_body={}
)
expect_error(status, body, 422, "invalid_query", "PATCH without changes")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items/999", method="PATCH", json_body={"cfg": 1.0}
)
expect_error(status, body, 404, "dataset_item_not_found", "PATCH unknown item")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH", json_body={"item_ids": [1, 2], "cfg": 6.0},
)
check(status == 200 and body == {"updated": 2}, "PATCH bulk updates")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items", method="PATCH", json_body={"item_ids": []}
)
expect_error(status, body, 422, "validation_error", "PATCH bulk empty ids")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items/discard",
    method="POST", json_body={"item_ids": [4]},
)
check(status == 200 and body == {"deleted": 1}, "discard removes a row")

# -- sets ------------------------------------------------------------------

status, _, body = asgi_request(app, "/api/v1/datasets/flow/sets")
check(status == 200 and body == {"sets": [], "count": 0}, "no sets yet")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/sets",
    method="POST", json_body={"name": "main", "item_ids": [1, 2]},
)
check(status == 201 and body["set_name"] == "main" and body["added"] == 2
      and body["set_id"] >= 1, "POST sets commits membership (201)")

status, _, body = asgi_request(app, "/api/v1/datasets/flow/sets")
check(body["count"] == 1 and body["sets"][0]["members"] == 2, "set listed with members")
status, _, body = asgi_request(app, "/api/v1/datasets/flow/items?committed=true")
check(body["count"] == 2, "committed filter after commit")

# -- tasks -----------------------------------------------------------------

status, _, body = asgi_request(app, "/api/v1/settings")
check(status == 200, "settings read for the checkpoints dir")
checkpoints = Path(body["resolved"]["checkpoints_dir"])
checkpoints.mkdir(parents=True, exist_ok=True)
(checkpoints / "m.safetensors").write_bytes(b"st")
imgs = root_b / "imgs"
imgs.mkdir()
for i in range(2):
    (imgs / f"{i}.png").write_bytes(b"png")

payload = {
    "kind": "ingest_lora",
    "image_dir": str(imgs),
    "model": "m.safetensors",
}
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks", method="POST", json_body=payload
)
check(status == 201 and body["status"] == "running" and body["pid"] == 7777
      and body["total"] == 2 and body["dataset"] == "flow",
      "POST tasks spawns (fake gateway) and records pid/count (201)")
task_id = body["id"]

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks", method="POST", json_body=payload
)
expect_error(status, body, 409, "dataset_task_active", "second task while active")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks",
    method="POST", json_body=dict(payload, kind="teacher"),
)
expect_error(status, body, 422, "invalid_query", "unknown task kind")

status, _, body = asgi_request(app, "/api/v1/datasets/flow/tasks")
check(status == 200 and body["count"] == 1 and body["tasks"][0]["id"] == task_id,
      "GET tasks lists the active one")

status, _, body = asgi_request(app, "/api/v1/datasets/flow")
check(body["active_tasks"][0]["id"] == task_id, "detail carries the active task")

status, _, body = asgi_request(app, "/api/v1/datasets/flow", method="DELETE")
expect_error(status, body, 409, "dataset_task_active", "delete while task active")

status, _, body = asgi_request(
    app, f"/api/v1/datasets/flow/tasks/{task_id}/stop", method="POST"
)
check(status == 200 and body["status"] == "killed", "stop kills the task")

status, _, body = asgi_request(
    app, f"/api/v1/datasets/flow/tasks/{task_id}/stop", method="POST"
)
expect_error(status, body, 409, "dataset_task_not_active", "stop of a killed task")

status, _, body = asgi_request(app, "/api/v1/datasets/flow", method="DELETE")
check(status == 200 and body == {"deleted": True}, "delete after stop succeeds")
status, _, body = asgi_request(app, "/api/v1/datasets/flow")
expect_error(status, body, 404, "dataset_not_found", "detail after delete")

status, _, body = asgi_request(app, "/api/v1/assets/dataset")
check({o["value"] for o in body["options"]} == {"made"},
      "catalog reflects the deletion")

finish()
