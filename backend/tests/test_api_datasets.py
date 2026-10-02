"""Dataset API tests -- end-to-end over raw ASGI.

Two wirings: the real ``build_container`` (read paths + error envelopes
+ the catalog-only assets kind) and ``build_services`` (the fake fork
gateway, so task lifecycle endpoints never spawn a child).

Run directly: python backend/tests/test_api_datasets.py
"""

from __future__ import annotations

import sqlite3
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
# Changed from no-cache: previews are per-dataset user data that changes
# when an item is re-previewed, so a shared cache must not keep them and
# a reload should not re-download unchanged bytes (N-05).
check(headers.get("cache-control") == "private, max-age=60",
      f"preview is private and briefly cacheable (got {headers.get('cache-control')!r})")

# Hardening headers: this route serves user-supplied bytes on the app's
# own origin, so the browser must not be allowed to reinterpret them.
check(headers.get("x-content-type-options") == "nosniff",
      f"nosniff (got {headers.get('x-content-type-options')!r})")
check("sandbox" in headers.get("content-security-policy", ""),
      f"CSP sandbox (got {headers.get('content-security-policy')!r})")
check("default-src 'none'" in headers.get("content-security-policy", ""),
      "CSP denies every source, not just scripts")
check("content-disposition" in {k.lower() for k in headers},
      f"content-disposition present (got {sorted(headers)})")

status, _, body = asgi_request(app, "/api/v1/datasets/api-ds/files/previews/none.png")
expect_error(status, body, 404, "dataset_file_not_found", "GET missing preview")

status, _, body = asgi_request(app, "/api/v1/datasets/nope/files/previews/p1.png")
expect_error(status, body, 404, "dataset_not_found", "GET preview, unknown dataset")

status, _, body = asgi_request(app, "/api/v1/datasets/api-ds/files/../../escape.txt")
expect_error(status, body, 404, "dataset_file_not_found", "traversal refused as not-found")
check("SECRET" not in str(body), "escape file never read")

# --- the allowlist (docs 08 N-05) -------------------------------------
# This route is the items grid's <img>. It used to serve every file in
# the dataset directory, so metadata.db, whole .safetensors shards (read
# into memory) and .svg as active content on this origin all came back
# 200. All three must now be indistinguishable from a missing file.
ds_root = root_a / "datasets" / "api-ds"
# Not the dataset's real metadata.db -- that is a live sqlite file this
# suite's own catalog listing opens, and writing junk into it breaks
# every later case for an unrelated reason. Same shape (a non-image file
# sitting inside the dataset directory), different name.
(ds_root / "notes.db").write_bytes(b"SQLite format 3\x00" + b"\0" * 4096)
(ds_root / "shards").mkdir(parents=True, exist_ok=True)
(ds_root / "shards" / "s0.safetensors").write_bytes(b"\0" * 5_000_000)
(ds_root / "previews" / "x.svg").write_text(
    '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(document.domain)"/>'
)
(ds_root / "previews" / "anim.gif").write_bytes(b"GIF89a")
# An oversized preview: over the cap, and must be refused before the read.
(ds_root / "previews" / "huge.png").write_bytes(
    b"\x89PNG\r\n\x1a\n" + b"\0" * (32 * 1024 * 1024)
)

for rel, label in (
    ("notes.db", "a non-image file inside the dataset dir"),
    ("shards/s0.safetensors", "a multi-MB safetensors shard"),
    ("previews/x.svg", "an svg (executes script on this origin)"),
    ("previews/anim.gif", "a gif"),
    ("previews/huge.png", "an oversized png"),
):
    status, _, body = asgi_request(app, f"/api/v1/datasets/api-ds/files/{rel}")
    expect_error(
        status, body, 404, "dataset_file_not_found",
        f"non-preview refused: {label}",
    )
    check("SQLite format" not in str(body) and "alert(" not in str(body),
          f"its bytes never reach the client ({label})")

# The allowed set still works, including case-insensitivity, and a real
# jpg/webp get their own media types.
(ds_root / "previews" / "p1.JPG").write_bytes(b"\xff\xd8\xff\xe0JFIF")
(ds_root / "previews" / "p1.webp").write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
for rel, media in (("previews/p1.JPG", "image/jpeg"), ("previews/p1.webp", "image/webp")):
    status, headers, _ = asgi_request(app, f"/api/v1/datasets/api-ds/files/{rel}")
    check(status == 200 and headers.get("content-type", "").startswith(media),
          f"{rel} serves as {media} (got {status} {headers.get('content-type')!r})")

# A symlink out of the tree stays refused, now for two reasons at once:
# it resolves outside the dataset root, and its suffix is checked too.
outside = root_a / "outside.png"
outside.write_bytes(b"\x89PNG\r\n\x1a\n" + b"secret")
link = ds_root / "previews" / "link.png"
if not link.exists():
    link.symlink_to(outside)
status, _, body = asgi_request(app, "/api/v1/datasets/api-ds/files/previews/link.png")
expect_error(status, body, 404, "dataset_file_not_found", "symlink outside refused")
check("secret" not in str(body), "the symlink target's bytes never arrive")

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

# A directory that is not a dataset is never deleted to make room for one
# (docs 07 F-10).
foreign = root_b / "datasets" / "my-images"
foreign.mkdir(parents=True, exist_ok=True)
(foreign / "IMG_0001.png").write_bytes(b"\x89PNG")
status, _, body = asgi_request(
    app, "/api/v1/datasets", method="POST", json_body={"name": "my-images"}
)
expect_error(
    status, body, 409, "dataset_directory_conflict", "create over a foreign directory"
)
check(
    (foreign / "IMG_0001.png").exists(),
    "the user's file is still there after the refusal",
)
check(
    not (foreign / "metadata.db").exists(),
    "and nothing was written into their directory",
)

status, _, body = asgi_request(app, "/api/v1/datasets/made")
check(status == 200 and body["info"]["description"] == "via api"
      and body["stats"]["items"] == 0 and body["sets"] == [] and body["active_tasks"] == [],
      "GET detail round-trip (identity + zeroed stats + empty sets/tasks)")

make_v2_dataset(root_b, "flow", items=4)

# -- items -----------------------------------------------------------------

status, _, body = asgi_request(app, "/api/v1/datasets/flow/items")
check(status == 200 and body["count"] == 4, "list items")

# Paged by default (docs 07 F-14, docs 08 Q10). This used to assert
# `limit is None` -- the "return everything" default -- which is exactly
# what cannot stand: a dataset grows by ingestion, so the unpaged
# response grew without bound and the UI could not tell a truncated list
# from the whole set. What matters now is that the response says which
# window it served and what was left over.
from backend.application.limits import DEFAULT_DATASET_ITEM_PAGE_SIZE  # noqa: E402

check(
    body["limit"] == DEFAULT_DATASET_ITEM_PAGE_SIZE and body["offset"] == 0,
    f"the default window is the default page size "
    f"(got limit={body['limit']}, offset={body['offset']})",
)
check(
    body["total"] == 4,
    f"total describes the whole dataset, not the page (got {body['total']})",
)
check(
    body["next_offset"] is None,
    f"a dataset smaller than a page has no next page (got {body['next_offset']})",
)
status, _, page = asgi_request(app, "/api/v1/datasets/flow/items?limit=2")
check(
    status == 200 and page["count"] == 2 and page["limit"] == 2
    and [item["id"] for item in page["items"]] == [1, 2],
    f"limit pages the rows (got {[i['id'] for i in page['items']]})",
)
check(
    page["total"] == 4 and page["next_offset"] == 2,
    f"a non-final page says where the next one starts "
    f"(got total={page['total']}, next_offset={page['next_offset']})",
)
status, _, tail = asgi_request(app, "/api/v1/datasets/flow/items?limit=2&offset=2")
check(
    [item["id"] for item in tail["items"]] == [3, 4],
    f"offset continues the page (got {[i['id'] for i in tail['items']]})",
)
# Rows 3-4 are the last of four, so this page ends the result.
check(
    tail["total"] == 4 and tail["next_offset"] is None,
    f"the final page reports no next (got total={tail['total']}, "
    f"next_offset={tail['next_offset']})",
)
status, _, end = asgi_request(app, "/api/v1/datasets/flow/items?limit=2&offset=4")
check(
    end["items"] == [] and end["next_offset"] is None,
    f"past the end is an empty page with no next "
    f"(got {end['items']}, {end['next_offset']})",
)
status, _, err = asgi_request(app, "/api/v1/datasets/flow/items?limit=501")
expect_error(
    status, err, 422, "validation_error", "an absurd page size is refused"
)
status, _, err = asgi_request(app, "/api/v1/datasets/flow/items?offset=-1")
expect_error(
    status, err, 422, "validation_error", "a negative offset is refused"
)
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

# -- multi-edit bulk (M8e): append/prepend/neg/verdict + guards --------

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH",
    json_body={"item_ids": [1, 2], "prompt": "trigger", "prompt_mode": "append"},
)
check(status == 200 and body == {"updated": 2}, "bulk append mode accepted")
status, _, body = asgi_request(app, "/api/v1/datasets/flow/items")
by_id = {i["id"]: i for i in body["items"]}
check(by_id[1]["prompt"].endswith("trigger") and by_id[2]["prompt"].endswith("trigger"),
      "append lands after the existing caption")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH",
    json_body={"item_ids": [1], "prompt": "trigger", "prompt_mode": "append"},
)
status, _, body = asgi_request(app, "/api/v1/datasets/flow/items")
by_id = {i["id"]: i for i in body["items"]}
check(by_id[1]["prompt"].count("trigger") == 1, "append is idempotent (no doubling)")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH",
    json_body={"item_ids": [1, 2], "neg_prompt": "lowres", "neg_prompt_mode": "set"},
)
check(status == 200 and body == {"updated": 2}, "bulk neg_prompt set accepted")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH",
    json_body={"item_ids": [1], "neg_prompt": ", ugly", "neg_prompt_mode": "append"},
)
status, _, body = asgi_request(app, "/api/v1/datasets/flow/items")
by_id = {i["id"]: i for i in body["items"]}
check(by_id[1]["neg_prompt"] == "lowres, ugly", "neg_prompt append composes with set")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH", json_body={"item_ids": [1, 2, 3], "type": "bad"},
)
check(status == 200 and body == {"updated": 3}, "bulk verdict flip accepted")
status, _, body = asgi_request(app, "/api/v1/datasets/flow/items")
check(all(i["type"] == "bad" for i in body["items"]),
      "bulk verdict landed on every selected row")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH", json_body={"item_ids": [1], "type": "meh"},
)
expect_error(status, body, 422, "invalid_query", "bulk verdict validated")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH",
    json_body={"item_ids": [1], "prompt": "x", "prompt_mode": "bogus"},
)
expect_error(status, body, 422, "invalid_query", "bulk prompt_mode validated")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items",
    method="PATCH",
    json_body={"item_ids": [1], "neg_prompt_mode": "bogus", "neg_prompt": "y"},
)
expect_error(status, body, 422, "invalid_query", "bulk neg_prompt_mode validated")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/items", method="PATCH", json_body={"item_ids": [1]}
)
expect_error(status, body, 422, "invalid_query", "bulk with no changes refused")

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

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks",
    method="POST", json_body=dict(payload, resize_mode="banana"),
)
expect_error(
    status, body, 422, "invalid_query",
    "unknown resize_mode refused -- 422 beats 409 even while active",
)

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

# -- generate_teacher (M8e) ---------------------------------------------

teacher_payload = {
    "kind": "generate_teacher",
    "model": "m.safetensors",
    "prompt_mode": "list",
    "prompts": "a red cube\na blue sphere",
    "negative_prompt": "blurry",
    "cfg_min": 3.0,
    "cfg_max": 9.0,
    "steps_min": 10,
    "steps_max": 20,
    "t_mode": "logit",
    "t_low": 20,
    "t_high": 999,
    "n_conditions": 5,
    "n_samples_per_cond": 3,
    "latent_size": 64,
    "model_type": "vpred",
    "seed": 7,
    "batch_size": 2,
}
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks", method="POST", json_body=teacher_payload
)
check(status == 201, f"generate_teacher accepted (201, got {status})")
if status == 201:
    check(
        body["kind"] == "generate_teacher" and body["total"] == 15
        and body["status"] == "running" and isinstance(body["pid"], int),
        f"teacher row: kind/total/status/pid "
        f"({body['kind']}, {body['total']}, {body['status']}, {body['pid']})",
    )
    check(
        body["params"]["model"] == "m.safetensors"
        and body["params"]["prompts"].startswith("a red cube")
        and body["params"]["model_type"] == "vpred"
        and body["params"]["n_conditions"] == 5,
        "teacher params record carries the launch payload flat",
    )
    teacher_id = body["id"]
else:
    teacher_id = None
    check(False, f"teacher launch body: {body}")

status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks",
    method="POST",
    json_body=dict(teacher_payload, steps_min=40, steps_max=10),
)
expect_error(status, body, 422, "invalid_query", "inverted steps range refused")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks",
    method="POST", json_body=dict(teacher_payload, prompts="\n  \n"),
)
expect_error(status, body, 422, "invalid_query", "empty prompt list refused")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks",
    method="POST", json_body=dict(teacher_payload, t_mode="quantum"),
)
expect_error(status, body, 422, "invalid_query", "unknown t_mode refused")
status, _, body = asgi_request(
    app, "/api/v1/datasets/flow/tasks",
    method="POST",
    json_body=dict(teacher_payload, prompt_mode="keywords", prompts=""),
)
expect_error(
    status, body, 422, "invalid_query",
    "keywords mode without a keyword source refused",
)
status, _, body = asgi_request(app, "/api/v1/datasets/flow/tasks")
check(
    status == 200 and body["count"] == 1 and body["tasks"][0]["id"] == teacher_id,
    "failed teacher launches never leave a row behind",
)

status, _, body = asgi_request(
    app, f"/api/v1/datasets/flow/tasks/{teacher_id}/stop", method="POST"
)
check(status == 200 and body["status"] == "killed", "teacher task stoppable")

status, _, body = asgi_request(app, "/api/v1/datasets/flow", method="DELETE")
check(status == 200 and body == {"deleted": True}, "delete after stop succeeds")
status, _, body = asgi_request(app, "/api/v1/datasets/flow")
expect_error(status, body, 404, "dataset_not_found", "detail after delete")

status, _, body = asgi_request(app, "/api/v1/assets/dataset")
check({o["value"] for o in body["options"]} == {"made"},
      "catalog reflects the deletion")

# ==========================================================================
# Section C: dataset card previews (M8f)
# ==========================================================================

root_c = Path(tempfile.mkdtemp(prefix="backend-api-ds-c-"))
services = build_services(project_root=root_c)
app = create_app(services)

make_v2_dataset(root_c, "cards", items=4)  # item 1 previews p1.png, item 4 is bad

# -- resolution: fallback when nothing is stored --------------------------

status, _, body = asgi_request(app, "/api/v1/datasets")
entry = next(d for d in body["datasets"] if d["info"]["name"] == "cards")
check(entry["preview_path"] == "previews/p1.png",
      "list entry falls back to the first non-bad item's preview")
status, _, body = asgi_request(app, "/api/v1/datasets/cards")
check(body["preview_path"] == "previews/p1.png",
      "detail carries the same resolved preview")

status, _, body = asgi_request(
    app, "/api/v1/datasets", method="POST", json_body={"name": "bare"}
)
check(status == 201, "empty dataset created")
status, _, body = asgi_request(app, "/api/v1/datasets/bare")
check(body["preview_path"] is None,
      "empty dataset previews to null (never fabricated)")

# -- set by item id -------------------------------------------------------

status, _, body = asgi_request(
    app, "/api/v1/datasets/cards/preview", method="PUT", json_body={"item_id": 1}
)
check(status == 200 and body == {"preview_path": "previews/p1.png"},
      f"PUT preview by item id (got {status} {body})")

# give item 2 its own preview file, then point the card at it
cards_dir = root_c / "datasets" / "cards"
(cards_dir / "previews" / "p2.png").write_bytes(b"\x89PNG")
conn = sqlite3.connect(str(cards_dir / "metadata.db"))
try:
    conn.execute(
        "UPDATE trajectories SET preview_path = 'previews/p2.png' WHERE id = 2"
    )
    conn.commit()
finally:
    conn.close()

status, _, body = asgi_request(
    app, "/api/v1/datasets/cards/preview", method="PUT", json_body={"item_id": 2}
)
check(status == 200 and body["preview_path"] == "previews/p2.png",
      "second PUT stores the override (upsert path)")
status, _, body = asgi_request(app, "/api/v1/datasets")
by_name = {d["info"]["name"]: d for d in body["datasets"]}
check(by_name["cards"]["preview_path"] == "previews/p2.png",
      "stored override wins over the fallback (list)")
status, _, body = asgi_request(app, "/api/v1/datasets/cards")
check(body["preview_path"] == "previews/p2.png",
      "stored override wins over the fallback (detail)")

# -- stale override (item discarded, file gone) degrades honestly ---------

(cards_dir / "previews" / "p2.png").unlink()
status, _, body = asgi_request(app, "/api/v1/datasets/cards")
check(body["preview_path"] == "previews/p1.png",
      "stale override falls back to a live preview instead of a dead path")
(cards_dir / "previews" / "p2.png").write_bytes(b"\x89PNG")  # restore fixture

# -- refusals -------------------------------------------------------------

status, _, body = asgi_request(
    app, "/api/v1/datasets/cards/preview", method="PUT", json_body={"item_id": 4}
)
expect_error(status, body, 422, "invalid_query",
             "preview refused on an item without a preview image")

status, _, body = asgi_request(
    app, "/api/v1/datasets/cards/preview", method="PUT", json_body={"item_id": 99}
)
expect_error(status, body, 404, "dataset_item_not_found",
             "preview refused for an unknown item")

status, _, body = asgi_request(
    app, "/api/v1/datasets/missing/preview", method="PUT", json_body={"item_id": 1}
)
expect_error(status, body, 404, "dataset_not_found",
             "preview refused on an unknown dataset")

status, _, body = asgi_request(
    app, "/api/v1/datasets/bare/preview", method="PUT", json_body={"item_id": 0}
)
expect_error(status, body, 422, "validation_error", "item_id must be >= 1")

status, _, body = asgi_request(
    app, "/api/v1/datasets/bare/preview", method="PUT", json_body={}
)
expect_error(status, body, 422, "validation_error", "item_id is required")

# -- legacy datasets: honest null + 409 on set ----------------------------

make_v1_dataset(root_c, "oldv1")
status, _, body = asgi_request(app, "/api/v1/datasets")
by_name = {d["info"]["name"]: d for d in body["datasets"]}
check(by_name["oldv1"]["preview_path"] is None,
      "legacy dataset previews to null (no v2 columns touched)")
status, _, body = asgi_request(
    app, "/api/v1/datasets/oldv1/preview", method="PUT", json_body={"item_id": 1}
)
expect_error(status, body, 409, "dataset_not_migrated",
             "preview set refused on a legacy dataset")

# -- delete drops the override (no inheritance by a same-name successor) --

status, _, body = asgi_request(app, "/api/v1/datasets/cards", method="DELETE")
check(status == 200 and body == {"deleted": True}, "cards deleted with its override")
make_v2_dataset(root_c, "cards", items=4)  # fresh dir, only previews/p1.png
status, _, body = asgi_request(app, "/api/v1/datasets/cards")
check(body["preview_path"] == "previews/p1.png",
      "recreated dataset resolves its own items, not the stale override")

finish()
