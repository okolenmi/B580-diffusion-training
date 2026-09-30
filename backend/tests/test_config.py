"""Config domain tests -- real ConfigFiles/ConfigInspector/Options
adapters over temp files, plus the API surface through build_services.

Run directly: python backend/tests/test_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import (
    ConfigInvalidError,
    ConfigNotFoundError,
    InvalidQueryError,
)
from backend.application.use_cases import (
    GetConfig,
    GetConfigOptions,
    GetStartOptions,
    ReadConfigRaw,
    UpdateConfig,
    WriteConfigRaw,
)
from backend.infrastructure.config_options import PydanticConfigOptions
from backend.infrastructure.core_config_files import CoreConfigFiles
from backend.infrastructure.core_config_inspector import CoreConfigInspector
from backend.infrastructure.workspace import WorkspaceLayout
from backend.presentation.app import create_app
from backend.tests.support import (
    build_services,
    check,
    finish,
    asgi_request,
    seed_run,
)

LORA_TOML = """
[common]
steps = 321
batch_size = 2

[paths]
base_model = "teacher_model.safetensors"
student = "student_model.safetensors"

[tuning]
method = "lora"
rank = 32
"""

DISTILL_TOML = """
[common]
steps = 500

[paths]
base_model = "teacher_model.safetensors"
resume_checkpoint = "/nonexistent/resume.safetensors"
"""


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="cfg-test-"))
    files = CoreConfigFiles()

    # -- read -----------------------------------------------------------
    cfg = _write(root, "cfg/test.toml", LORA_TOML)
    read = GetConfig(files=files, project_root=root)
    data = read.execute("cfg/test.toml")
    check(data["common"]["steps"] == 321, "read returns nested config")
    check(data["tuning"]["method"] == "lora", "read carries union variant")
    check(data["tuning"]["rank"] == 32, "read carries variant field")

    try:
        read.execute("")
        check(False, "empty path rejected")
    except InvalidQueryError:
        check(True, "empty path rejected")

    try:
        read.execute("cfg/nope.toml")
        check(False, "missing config rejected")
    except ConfigNotFoundError:
        check(True, "missing config rejected")

    bad = _write(root, "cfg/bad.toml", "[common]\nsteps = 3\n")
    try:
        read.execute("cfg/bad.toml")
        check(False, "invalid config rejected")
    except ConfigInvalidError:
        check(True, "invalid config rejected (steps below ge=100)")

    # -- update (deep merge) --------------------------------------------
    update = UpdateConfig(files=files, project_root=root)
    merged = update.execute("cfg/test.toml", {"common": {"batch_size": 4}})
    check(merged["common"]["batch_size"] == 4, "update applies override")
    check(merged["common"]["steps"] == 321, "update preserves sibling fields")
    check(merged["tuning"]["rank"] == 32, "update preserves other sections")
    reread = read.execute("cfg/test.toml")
    check(reread["common"]["batch_size"] == 4, "update persisted to disk")

    before = (root / "cfg" / "bad.toml").read_text()
    try:
        update.execute("cfg/bad.toml", {"common": {"steps": 1}})
        check(False, "invalid update rejected")
    except ConfigInvalidError:
        check(True, "invalid update rejected")
    check(
        (root / "cfg" / "bad.toml").read_text() == before,
        "rejected update left the file untouched",
    )

    try:
        update.execute("cfg/missing.toml", {"common": {"steps": 101}})
        check(False, "update of missing file rejected")
    except ConfigNotFoundError:
        check(True, "update of missing file rejected")

    # Unknown keys are ignored (TrainingConfig is permissive by design).
    tolerated = update.execute("cfg/test.toml", {"not_a_section": {"x": 1}})
    check(tolerated["common"]["steps"] == 321, "unknown override keys ignored")

    # -- raw -------------------------------------------------------------
    raw = ReadConfigRaw(files=files, project_root=root)
    content = raw.execute("cfg/test.toml").content
    check("steps = 321" in content or "steps=321" in content, "raw read returns text")

    write_raw = WriteConfigRaw(files=files, project_root=root)
    write_raw.execute("cfg/new.toml", "[common]\nsteps = 777\n")
    check(read.execute("cfg/new.toml")["common"]["steps"] == 777, "raw write creates file")

    try:
        write_raw.execute("cfg/new.toml", "[common]\nsteps = 1\n")
        check(False, "invalid raw document rejected")
    except ConfigInvalidError:
        check(True, "invalid raw document rejected")
    check(
        read.execute("cfg/new.toml")["common"]["steps"] == 777,
        "rejected raw write left the file unchanged",
    )

    # -- options schema (pure) -------------------------------------------
    options = GetConfigOptions(options=PydanticConfigOptions())
    schema = options.execute()
    ids = [opt["id"] for opt in schema]
    check(len(schema) > 40, "schema has entries")
    check(ids[:3] == ["start_from", "start_from", "reset_optimizer"],
          "synthetic launch options come first")
    steps_opt = next(o for o in schema if o["id"] == "common.steps")
    check(steps_opt["type"] == "number", "steps is a number")
    check(steps_opt["min"] == 100 and steps_opt["max"] == 200000, "steps bounds derived")
    check(steps_opt["label"] == "Total Steps", "hand-authored label merged")
    check(not any(i.startswith("preview.") or i == "preview" for i in ids),
          "preview section absent from the schema")
    rank_opt = next(o for o in schema if o["id"] == "tuning.rank")
    check(rank_opt.get("visible_when", {}).get("tuning.method") == "lora",
          "union-variant field tagged with its variant")
    base_opt = next(o for o in schema if o["id"] == "paths.base_model")
    check(base_opt.get("file_kind") == "checkpoint", "file_kind metadata merged")
    method_opt = next(o for o in schema if o["id"] == "tuning.method")
    check(method_opt["choices"][0]["value"] == "distillation",
          "choice order curated")

    # -- describe (real inspector, absolute paths) ------------------------
    layout = WorkspaceLayout(root, runs_dir=root / "runs")
    inspector = CoreConfigInspector(layout)
    lora_file = root / "made_lora.safetensors"
    lora_file.write_bytes(b"\x00")
    teacher_file = root / "teacher_model.safetensors"
    teacher_file.write_bytes(b"\x00")

    lora_cfg = _write(
        root,
        "cfg/lora.toml",
        LORA_TOML.replace("teacher_model.safetensors", str(teacher_file)),
    )
    described = inspector.describe(lora_cfg)
    check(described.mode == "lora", "describe mode")
    check("lora_checkpoint" in described.start_from, "lora config offers lora_checkpoint")
    check(described.start_from["teacher"].available, "existing absolute teacher available")
    check(described.start_from["student"].path == "student_model.safetensors",
          "describe reports configured path as-is")
    check(not described.start_from["student"].available, "missing student unavailable")
    check(not described.start_from["lora_checkpoint"].available,
          "unset lora output unavailable")

    distill_cfg = _write(
        root,
        "cfg/distill.toml",
        DISTILL_TOML.replace("/nonexistent/resume.safetensors", str(root / "nope.safetensors")),
    )
    described = inspector.describe(distill_cfg)
    check("lora_checkpoint" not in described.start_from,
          "non-lora config omits lora_checkpoint entirely (absent, not false)")
    check(not described.start_from["resume"].available, "missing resume unavailable")

    # -- API surface -------------------------------------------------------
    from backend.tests.support import FakeClock, InMemoryRunRepository

    runs = InMemoryRunRepository()
    clock = FakeClock()
    services = build_services(runs=runs, clock=clock, project_root=root)
    app = create_app(services)

    status, _, body = asgi_request(app, "/api/v1/config?path=cfg/test.toml")
    check(status == 200 and body["common"]["steps"] == 321, "GET config 200")

    status, _, body = asgi_request(app, "/api/v1/config")
    check(status == 422 and body["error"]["code"] == "invalid_query",
          "GET config without path -> 422 invalid_query")

    status, _, body = asgi_request(app, "/api/v1/config?path=cfg/none.toml")
    check(status == 404 and body["error"]["code"] == "config_not_found",
          "GET missing config -> 404 envelope")

    status, _, body = asgi_request(
        app, "/api/v1/config", method="PATCH",
        json_body={"path": "cfg/test.toml", "overrides": {"common": {"lr": 0.001}}},
    )
    check(status == 200 and body["common"]["lr"] == 0.001, "PATCH config 200")
    check(body["common"]["steps"] == 321, "PATCH merges, not replaces")

    status, _, body = asgi_request(
        app, "/api/v1/config", method="PATCH",
        json_body={"path": "cfg/bad.toml", "overrides": {"common": {"steps": 7}}},
    )
    check(status == 422 and body["error"]["code"] == "config_invalid",
          "PATCH invalid -> 422 config_invalid envelope")

    status, _, body = asgi_request(app, "/api/v1/config/raw?path=cfg/test.toml")
    check(status == 200 and "[common]" in body["content"], "GET raw config")

    status, _, body = asgi_request(
        app, "/api/v1/config/raw", method="PUT",
        json_body={"path": "cfg/created.toml", "content": "[common]\nsteps = 654\n"},
    )
    check(status == 200 and body["ok"] is True, "PUT raw creates config")

    status, _, body = asgi_request(app, "/api/v1/config/options")
    check(status == 200 and len(body["options"]) > 40, "GET options schema")

    # start-options through the API (fake inspector, scripted description)
    status, _, body = asgi_request(app, "/api/v1/config/start-options?path=cfg/test.toml")
    check(status == 200, "GET start-options 200")
    check(body["has_unfinished_run"] is False, "no active run -> false")
    check(body["last_finished"] is None, "no finished run -> null")
    check("teacher" in body["start_from"], "start_from map present")
    check("lora_checkpoint" not in body["start_from"],
          "fake description's absent key stays absent over HTTP")

    seed_run(runs, clock, start=True)
    status, _, body = asgi_request(app, "/api/v1/config/start-options?path=cfg/test.toml")
    check(body["has_unfinished_run"] is True, "running run -> has_unfinished_run true")

    seed = seed_run(runs, clock, start=True)
    seed.mark_completed(at=clock.now())
    runs.update(seed)
    status, _, body = asgi_request(app, "/api/v1/config/start-options?path=cfg/test.toml")
    check(
        body["last_finished"] is not None
        and body["last_finished"]["id"] == seed.id
        and body["last_finished"]["status"] == "completed",
        "last_finished surfaces the newest finished run",
    )

    finish()


if __name__ == "__main__":
    main()
