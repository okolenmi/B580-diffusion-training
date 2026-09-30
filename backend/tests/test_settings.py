"""Settings domain tests -- store persistence/validation, resolution
tiers, and the API surface.

Run directly: python backend/tests/test_settings.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import SettingsInvalidError
from backend.application.ports.settings_store import SettingsChanges
from backend.infrastructure.persistence.sqlite import SqliteDatabase
from backend.infrastructure.settings_store import SqliteSettingsStore
from backend.infrastructure.workspace import WorkspaceLayout
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, build_services, check, finish


def _store(root: Path) -> SqliteSettingsStore:
    database = SqliteDatabase(root / "settings-test.db")
    database.initialize()
    return SqliteSettingsStore(database, root)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="settings-test-"))
    store = _store(root)

    # -- defaults ---------------------------------------------------------
    view = store.read()
    check(all(view.stored[k] == "" for k in view.stored), "defaults: nothing stored")
    check(set(view.resolved) == {"comfy_dir", "venv_python", "checkpoints_dir", "loras_dir"},
          "resolved exposes the four path settings")
    check(isinstance(view.resolved["venv_python"], str) and view.resolved["venv_python"],
          "venv_python always resolves")
    check(isinstance(view.resolved["checkpoints_dir"], str) and view.resolved["checkpoints_dir"],
          "checkpoints_dir always resolves")

    # -- persist + resolve -------------------------------------------------
    ckpt = root / "my-checkpoints"
    view = store.update(SettingsChanges(checkpoints_dir=str(ckpt), default_config="d.toml"))
    check(view.stored["checkpoints_dir"] == str(ckpt), "override persisted")
    check(view.resolved["checkpoints_dir"] == str(ckpt), "override wins resolution")
    check(ckpt.is_dir(), "managed dir created on save")
    check(view.stored["default_config"] == "d.toml", "default_config persisted")
    check(store.get("checkpoints_dir") == str(ckpt), "raw get returns stored value")

    # absent key = untouched
    view = store.update(SettingsChanges(default_config="other.toml"))
    check(view.stored["checkpoints_dir"] == str(ckpt), "absent key untouched by update")
    check(view.stored["default_config"] == "other.toml", "provided key updated")

    # empty string = clear
    view = store.update(SettingsChanges(checkpoints_dir=""))
    check(view.stored["checkpoints_dir"] == "", "empty string clears override")
    check(view.resolved["checkpoints_dir"] != str(ckpt),
          "cleared override falls back to auto-detection")

    # -- atomic validation --------------------------------------------------
    try:
        store.update(SettingsChanges(venv_python="/nonexistent/python-xyz",
                                     default_config="should-not-persist.toml"))
        check(False, "invalid venv_python rejected")
    except SettingsInvalidError as exc:
        check(True, "invalid venv_python rejected")
        check(isinstance(exc.details, dict) and "venv_python" in exc.details,
              "error carries per-key details")
    check(store.get("default_config") == "other.toml",
          "rejected update persisted nothing (atomic)")

    try:
        store.update(SettingsChanges(comfy_dir="/nonexistent/comfy-dir"))
        check(False, "invalid comfy_dir rejected")
    except SettingsInvalidError:
        check(True, "invalid comfy_dir rejected")

    # -- env tier beats stored override for venv -----------------------------
    saved_env = os.environ.get("VENV_PYTHON")
    try:
        os.environ["VENV_PYTHON"] = sys.executable
        view = store.update(SettingsChanges(venv_python="/bin/sh"))
        check(view.resolved["venv_python"] == sys.executable,
              "env VENV_PYTHON beats stored override")

        del os.environ["VENV_PYTHON"]
        view = store.read()
        check(view.resolved["venv_python"] == "/bin/sh",
              "stored override used once env is gone")

        # -- layout reads the store's tier (env stays cleared here) ----------
        view = store.update(SettingsChanges(checkpoints_dir=str(ckpt),
                                            venv_python=sys.executable))
        layout = WorkspaceLayout(root, runs_dir=root / "runs", settings_kv=store.get)
        check(layout.checkpoints_dir == ckpt, "layout honors stored checkpoints override")
        check(layout.venv_python == sys.executable, "layout honors stored venv_python")

        plain = WorkspaceLayout(root, runs_dir=root / "runs")  # no kv: env + defaults
        check(plain.checkpoints_dir != ckpt,
              "layout without settings_kv ignores overrides")
    finally:
        if saved_env is None:
            os.environ.pop("VENV_PYTHON", None)
        else:
            os.environ["VENV_PYTHON"] = saved_env

    # -- API surface ----------------------------------------------------------
    # Same store as above: the API must see what was persisted (fresh reads).
    services = build_services(project_root=root, settings_store=store)
    app = create_app(services)

    status, _, body = asgi_request(app, "/api/v1/settings")
    check(status == 200, "GET settings 200")
    check("stored" in body and "resolved" in body, "view has stored/resolved halves")
    check(body["stored"]["checkpoints_dir"] == str(ckpt),
          "API sees the persisted override (fresh read, no cache)")

    status, _, body = asgi_request(
        app, "/api/v1/settings", method="POST",
        json_body={"default_config": "from-api.toml"},
    )
    check(status == 200 and body["stored"]["default_config"] == "from-api.toml",
          "POST settings partial update 200")
    check(body["stored"]["checkpoints_dir"] == str(ckpt),
          "POST leaves absent keys untouched")

    status, _, body = asgi_request(
        app, "/api/v1/settings", method="POST",
        json_body={"venv_python": "/nonexistent/via-api"},
    )
    check(status == 400 and body["error"]["code"] == "settings_invalid",
          "POST invalid -> 400 settings_invalid envelope")
    check(body["error"]["details"]["venv_python"],
          "envelope details carry the per-key message")

    status, _, body = asgi_request(app, "/api/v1/settings")
    check(body["stored"]["default_config"] == "from-api.toml",
          "failed POST changed nothing")

    status, _, body = asgi_request(
        app, "/api/v1/settings", method="POST",
        json_body={"checkpoints_dir": ""},
    )
    check(status == 200 and body["stored"]["checkpoints_dir"] == "",
          "POST empty string clears the override")

    finish()


if __name__ == "__main__":
    main()
