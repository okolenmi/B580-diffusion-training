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
from backend.application.project_paths import ProjectPaths
from backend.application.use_cases import (
    GetConfig,
    GetConfigOptions,
    ReadConfigRaw,
    UpdateConfig,
    WriteConfigRaw,
)
from backend.infrastructure.config_options import PydanticConfigOptions
from backend.infrastructure.core_config_files import CoreConfigFiles
from backend.infrastructure.core_config_inspector import CoreConfigInspector
from backend.infrastructure.workspace import WorkspaceLayout
from backend.presentation.app import create_app
from backend.python_floor import (
    MESSAGE,
    MIN_PYTHON,
    require_python,
    require_python_at_least,
)
from backend.tests.support import (
    build_services,
    check,
    finish,
    asgi_request,
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
    paths = ProjectPaths(root=root)

    # -- read -----------------------------------------------------------
    _write(root, "cfg/test.toml", LORA_TOML)
    read = GetConfig(files=files, paths=paths)
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

    _write(root, "cfg/bad.toml", "[common]\nsteps = 3\n")
    try:
        read.execute("cfg/bad.toml")
        check(False, "invalid config rejected")
    except ConfigInvalidError:
        check(True, "invalid config rejected (steps below ge=100)")

    # -- update (deep merge) --------------------------------------------
    update = UpdateConfig(files=files, paths=paths)
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
    raw = ReadConfigRaw(files=files, paths=paths)
    content = raw.execute("cfg/test.toml").content
    check("steps = 321" in content or "steps=321" in content, "raw read returns text")

    write_raw = WriteConfigRaw(files=files, paths=paths)
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

    # The raw editor stores the user's own text (docs 07 F-08): comments,
    # key order and keys the model does not declare must survive a save.
    annotated = (
        "# tuning notes for the next attempt\n"
        "[common]\n"
        "steps = 888   # raised after the OOM\n"
        "\n"
        "[experimental]\n"
        "my_note = \"keep me\"\n"
    )
    write_raw.execute("cfg/annotated.toml", annotated)
    stored = (root / "cfg" / "annotated.toml").read_text(encoding="utf-8")
    check(stored == annotated, f"the file is byte-for-byte what was sent (got {stored!r})")
    check("# tuning notes" in stored, "the comment survived")
    check("steps = 888   # raised after the OOM" in stored, "the inline comment survived")
    check("[experimental]" in stored and "my_note" in stored, "the unknown table survived")
    check(
        read.execute("cfg/annotated.toml")["common"]["steps"] == 888,
        "and the model still reads it (validation ran on save)",
    )
    check(
        not (root / "cfg" / ".annotated.toml.partial").exists(),
        "the atomic temp file is gone after the write",
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
    services = build_services(project_root=root)
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

    # -- the raw routes stay inside the project --------------------------
    #
    # Found by sweeping every operation for 5xx and then following the
    # query-string dimension: ProjectPaths had no containment at all, only
    # a non-empty check, so a client-named path was used as sent.
    #
    #   GET  /api/v1/config/raw?path=../../../../etc/passwd  -> 200 + file
    #   PUT  ... the same path                                -> 200 + written
    #
    # Arbitrary read and arbitrary write, from a route meant to edit this
    # project's configuration. Both verbs are checked, and the write case
    # asserts nothing landed, because a 422 that still created the file
    # would be no fix at all.
    #
    # A NUL needs spelling differently per transport: percent-encoded in a
    # query string, where it decodes to one, and a real \x00 in a JSON body,
    # where "%00" is three ordinary characters and makes a perfectly legal
    # filename.
    escapes = [
        ("../../../../etc/passwd", "traversal out of the project"),
        ("/etc/passwd", "an absolute path outside the project"),
        ("cfg/../../escape.toml", "traversal that starts inside"),
    ]
    for escape, label in escapes:
        status, _, body = asgi_request(app, f"/api/v1/config/raw?path={escape}")
        check(status == 422 and body["error"]["code"] == "invalid_query",
              f"GET raw refuses {label} ({status} {body.get('error', {}).get('code')})")
        status, _, body = asgi_request(
            app, "/api/v1/config/raw", method="PUT",
            json_body={"path": escape, "content": "escaped = true\n"},
        )
        check(status == 422 and body["error"]["code"] == "invalid_query",
              f"PUT raw refuses {label} ({status} {body.get('error', {}).get('code')})")

    # Each transport needs its own spelling, and the mismatch is itself
    # worth stating: in a query string "%00" decodes to a NUL, while in a
    # JSON body it is three ordinary characters -- a perfectly legal
    # filename, and one the route is right to accept. Only a real \x00 in
    # the body is a NUL.
    status, _, body = asgi_request(app, "/api/v1/config/raw?path=cfg/%00.toml")
    check(status == 422 and body["error"]["code"] == "invalid_query",
          f"GET raw refuses a percent-encoded NUL ({status})")

    # Double-encoded is *not* a NUL: it decodes once to the three characters
    # "%00", which is a legal filename, so the honest answer is the ordinary
    # not-found for a file that is not there -- not the refusal above.
    status, _, body = asgi_request(app, "/api/v1/config/raw?path=cfg/%2500.toml")
    check(status == 404 and body["error"]["code"] == "config_not_found",
          f"a double-encoded NUL is an ordinary missing filename ({status} "
          f"{body.get('error', {}).get('code')})")

    for body_nul, label in (("\x00", "a real NUL"),
                            ("a%00b", "literal percent-zero-zero, a legal name")):
        status, _, body = asgi_request(
            app, "/api/v1/config/raw", method="PUT",
            json_body={"path": f"cfg/{body_nul}.toml", "content": "x = 1\n"},
        )
        expect = 422 if body_nul == "\x00" else 200
        check(status == expect,
              f"PUT raw on {label}: {status}, wanted {expect}")

    status, _, body = asgi_request(
        app, "/api/v1/config/raw", method="PUT",
        json_body={"path": "cfg/\x00.toml", "content": "x = 1\n"},
    )
    check(not (root / "cfg" / "\x00.toml").exists(),
          "and a NUL-named file was not created")
    check(not (root / "escape.toml").exists(),
          "nor anything from the traversal cases")

    # The two shapes that must keep working: a project-relative path, and an
    # absolute path that lands inside the project -- callers legitimately
    # hold one for a config they are about to write. Content is copied from
    # a config already known to validate, because this is a test about paths
    # and a rejected document would answer 422 for an unrelated reason.
    valid = (root / "cfg" / "test.toml").read_text(encoding="utf-8")
    status, _, body = asgi_request(app, "/api/v1/config/raw?path=cfg/test.toml")
    check(status == 200, f"a project-relative path still reads ({status})")
    status, _, body = asgi_request(
        app, "/api/v1/config/raw", method="PUT",
        json_body={"path": "cfg/relative-again.toml", "content": valid},
    )
    check(status == 200,
          f"a project-relative path still writes ({status} "
          f"{body.get('error', {}).get('code')})")
    status, _, body = asgi_request(
        app, "/api/v1/config/raw", method="PUT",
        json_body={"path": str(root / "cfg" / "absolute.toml"), "content": valid},
    )
    check(status == 200,
          f"an absolute path inside the project still writes ({status} "
          f"{body.get('error', {}).get('code')})")

    status, _, body = asgi_request(app, "/api/v1/config/options")
    check(status == 200 and len(body["options"]) > 40, "GET options schema")

    # The route that used to sit here was /config/start-options: the
    # "continue from" picker, which reported whether a run was active and
    # which was the last one finished. Both halves were about the
    # supervised-subprocess route and are gone with it (training starts
    # from a graph execution now). Its absence is asserted from
    # test_error_contract.py's route inventory.

    # -- the XPU-only optimizer choices (docs/decisions/0004-b580-only) ---
    #
    # `xpu-adafactor` exists because it behaves well on this project's one
    # supported device, and the Adafactor scale parameter only appears when
    # an Adafactor flavour is chosen. Nothing else in the suite covers this,
    # so a refactor of the UI metadata could quietly drop the XPU option and
    # leave the config page offering optimizers that are wrong for the
    # hardware. Written against the flat option list `execute()` returns,
    # keyed by `id` -- not the nested metadata table in config_ui_data.py,
    # which is an input to it rather than the thing the UI is driven by.
    print("\n== optimizer choices suit the device ==")

    by_id = {opt["id"]: opt for opt in schema}
    optimizer = by_id["common.optimizer"]
    values = [c["value"] for c in optimizer["choices"]]
    check("xpu-adafactor" in values,
          f"the XPU Adafactor choice is offered (got {values})")
    labelled = {c["value"]: c["label"] for c in optimizer["choices"]}
    check(labelled["xpu-adafactor"] == "XPU Adafactor",
          f"and is labelled as such rather than shown as a raw enum "
          f"(got {labelled['xpu-adafactor']!r})")

    scale_when = by_id["common.adafactor_scale_param"]["visible_when"]
    shown_for = scale_when.get("common.optimizer", [])
    check("xpu-adafactor" in shown_for,
          f"the scale parameter appears when it is chosen (got {shown_for})")
    check("fused-adafactor" in shown_for,
          "and when the fused flavour is chosen")
    check("adamw" not in shown_for,
          f"but not for AdamW, which has no scale parameter (got {shown_for})")

    # -- the interpreter floor is stated, not assumed -----------------------
    # Both quality tools were already configured for 3.14, which tells the
    # tools and not the person. A checkout on an older interpreter passed
    # both gates and then failed somewhere else entirely -- and the one
    # runtime dependency on the version, `get_origin(x) is Union` in
    # event_schema.py, fails *silently* there, producing a wrong schema
    # rather than an error. The floor is therefore stated once and checked
    # at both process entry points.
    check(sys.version_info >= MIN_PYTHON,
          f"this interpreter satisfies the declared floor {MIN_PYTHON} "
          f"(running {'.'.join(str(p) for p in sys.version_info[:3])})")
    check("3.14" in MESSAGE and "get_origin" in MESSAGE,
          "and the refusal names the version and the reason, so a developer "
          "is not left to work out which interpreter to install")

    require_python()   # and does not refuse this one

    # And the check itself refuses an old interpreter, rather than being a
    # comparison nobody has ever seen return the other way.
    raised = None
    try:
        require_python_at_least((99, 0))
    except SystemExit as exc:
        raised = str(exc)
    check(raised is not None and "99.0" in raised,
          f"an impossible floor is refused with a message naming it "
          f"(got {raised!r})")


    finish()


if __name__ == "__main__":
    main()
