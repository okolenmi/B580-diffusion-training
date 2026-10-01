"""Asset domain tests -- real FileSystemAssetStore over temp dirs,
sandboxing behavior, header-only inspect, and the API surface.

Run directly: python backend/tests/test_assets.py
"""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import AssetTooLargeError, InvalidQueryError
from backend.application.ports.asset_store import MAX_UPLOAD_BYTES
from backend.application.ports.settings_store import SettingsChanges
from backend.infrastructure.file_asset_store import FileSystemAssetStore
from backend.infrastructure.workspace import WorkspaceLayout
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, build_services, check, finish


def _layout(root: Path, ckpt: Path, loras: Path) -> WorkspaceLayout:
    overrides = {"checkpoints_dir": str(ckpt), "loras_dir": str(loras)}
    return WorkspaceLayout(
        root,
        runs_dir=root / "runs",
        settings_kv=lambda key, default: overrides.get(key, default),
    )


def _write_safetensors(path: Path, keys: dict[str, list[list[float]]]) -> None:
    """Minimal valid safetensors: 8-byte LE header length + JSON header + data."""
    header: dict = {}
    data = b""
    offset = 0
    for name, rows in keys.items():
        flat = [v for row in rows for v in row]
        raw = struct.pack(f"<{len(flat)}f", *flat)
        header[name] = {
            "dtype": "F32",
            "shape": [len(rows), len(rows[0])],
            "data_offsets": [offset, offset + len(raw)],
        }
        data += raw
        offset += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="assets-test-"))
    ckpt = root / "checkpoints"
    loras = root / "loras"
    (ckpt / "sub").mkdir(parents=True)
    (ckpt / "resume").mkdir()
    (ckpt / ".hidden_dir").mkdir()
    (ckpt / "a.safetensors").write_bytes(b"x")
    (ckpt / "sub" / "b.safetensors").write_bytes(b"x")
    (ckpt / "resume" / "r.safetensors").write_bytes(b"x")
    (ckpt / ".dot.safetensors").write_bytes(b"x")
    loras.mkdir()

    store = FileSystemAssetStore(_layout(root, ckpt, loras))

    # -- catalog -----------------------------------------------------------
    catalog = store.catalog("checkpoint")
    check(catalog.base_dir == str(ckpt), "catalog reports the resolved base dir")
    check(
        [o.value for o in catalog.options] == ["a.safetensors", "sub/b.safetensors"],
        "catalog lists nested files, excluding resume/ and dotfiles",
    )
    check(catalog.upload_supported and catalog.browse_supported, "capabilities reported")

    empty = store.catalog("lora")
    check(empty.options == (), "catalog of an empty/missing dir is empty, not an error")

    try:
        store.catalog("dataset")
        check(False, "unknown kind rejected")
    except InvalidQueryError:
        check(True, "unknown kind rejected")

    # -- browse ------------------------------------------------------------
    listing = store.browse("checkpoint")
    check(listing.folders == ("sub",), "browse folders exclude resume/ and dotdirs")
    check(listing.files == ("a.safetensors",), "browse files are immediate only")

    nested = store.browse("checkpoint", "sub")
    check(nested.files == ("b.safetensors",), "browse into a subfolder")

    gone = store.browse("checkpoint", "missing-dir")
    check(gone.folders == () and gone.files == (), "browse missing dir is empty")

    for bad in ("../escape", "/etc", "..\\escape"):
        try:
            store.browse("checkpoint", bad)
            check(False, f"sandbox rejects browse {bad!r}")
        except InvalidQueryError:
            check(True, f"sandbox rejects browse {bad!r}")

    try:
        store.browse("checkpoint", "a.safetensors")
        check(False, "browse of a file rejected")
    except InvalidQueryError:
        check(True, "browse of a file rejected")

    # -- writes ------------------------------------------------------------
    made = store.make_folder("checkpoint", "new/deep")
    check((ckpt / "new" / "deep").is_dir(), "make_folder creates nested folders")

    saved = store.save_upload("lora", "sub/up.safetensors", b"payload-bytes")
    check(Path(saved).read_bytes() == b"payload-bytes", "save_upload round-trips bytes")

    for bad in ("../evil.safetensors", "/abs.safetensors"):
        try:
            store.save_upload("lora", bad, b"x")
            check(False, f"sandbox rejects upload {bad!r}")
        except InvalidQueryError:
            check(True, f"sandbox rejects upload {bad!r}")

    # upload policy: extension allowlist + size cap, nothing written
    try:
        store.save_upload("lora", "evil.sh", b"#!/bin/sh")
        check(False, "upload rejects a non-safetensors extension")
    except InvalidQueryError:
        check(True, "upload rejects a non-safetensors extension")
    check(not (loras / "evil.sh").exists(), "extension-rejected upload writes nothing")

    store.max_upload_bytes = 4
    try:
        store.save_upload("lora", "big.safetensors", b"12345")
        check(False, "upload enforces the size cap")
    except AssetTooLargeError:
        check(True, "upload enforces the size cap")
    store.max_upload_bytes = MAX_UPLOAD_BYTES
    check(
        not (loras / "big.safetensors").exists()
        and not (loras / "big.safetensors.part").exists(),
        "size-rejected upload leaves no file and no .part",
    )
    check(not list(loras.rglob("*.part")), "successful upload leaves no .part behind")

    # -- inspect -----------------------------------------------------------
    good = ckpt / "good.safetensors"
    _write_safetensors(good, {"model.diffusion_model.blocks.0.weight": [[1.0, 2.0]]})
    info = store.inspect("checkpoint", "good.safetensors")
    check(info["kind"] == "checkpoint", "inspect contract: kind")
    check(info["path"] == "good.safetensors", "inspect contract: path as sent")
    unet = info["components"]["unet"]
    check(unet["key_count"] == 1 and unet["dtype"] == "float32",
          "inspect reports component dtype + key count from the header")

    corrupt = ckpt / "corrupt.safetensors"
    corrupt.write_bytes(b"not a safetensors file at all")
    try:
        store.inspect("checkpoint", "corrupt.safetensors")
        check(False, "corrupt file rejected")
    except InvalidQueryError:
        check(True, "corrupt file rejected")

    try:
        store.inspect("checkpoint", "nope.safetensors")
        check(False, "missing file rejected")
    except InvalidQueryError:
        check(True, "missing file rejected")

    lora_good = loras / "adapter.safetensors"
    _write_safetensors(
        lora_good,
        {
            "lora_unet_attn.lora_down.weight": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            "lora_unet_attn.lora_up.weight": [[0.5, 0.5, 0.5, 0.5]],
        },
    )
    lora_info = store.inspect("lora", "adapter.safetensors")
    check(
        lora_info["kind"] == "lora"
        and lora_info["rank"] == 2
        and lora_info["key_count"] == 1
        and lora_info["dtype"] == "float32",
        "LoRA inspect reports dtype/rank/key_count",
    )

    _write_safetensors(loras / "not_a_lora.safetensors", {"some.weight": [[1.0]]})
    not_lora = store.inspect("lora", "not_a_lora.safetensors")
    check(
        not_lora["rank"] is None and not_lora["key_count"] == 0,
        "valid safetensors without LoRA keys -> zero counts, no crash",
    )

    # -- API surface -------------------------------------------------------
    services = build_services(project_root=root)
    services.settings.update.execute(
        SettingsChanges(checkpoints_dir=str(ckpt), loras_dir=str(loras))
    )
    app = create_app(services)

    status, _, body = asgi_request(app, "/api/v1/assets/checkpoint")
    check(status == 200 and body["kind"] == "checkpoint", "GET assets catalog 200")
    check(
        body["files"]
        == ["a.safetensors", "corrupt.safetensors", "good.safetensors", "sub/b.safetensors"],
        "catalog files exposed for pickers",
    )
    check(body["base_dir"] == str(ckpt), "catalog base_dir reflects stored override")

    status, _, body = asgi_request(app, "/api/v1/assets/bogus")
    check(status == 422 and body["error"]["code"] == "invalid_query",
          "unknown kind -> 422 invalid_query envelope")

    status, _, body = asgi_request(app, "/api/v1/assets/checkpoint/browse?path=sub")
    check(status == 200 and body["files"] == ["b.safetensors"], "GET browse 200")

    status, _, body = asgi_request(app, "/api/v1/assets/checkpoint/browse?path=../x")
    check(status == 422 and body["error"]["code"] == "invalid_query",
          "sandbox violation over HTTP -> 422 envelope")

    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/inspect?path=good.safetensors"
    )
    check(status == 200 and body["components"]["unet"]["key_count"] == 1,
          "GET inspect 200")

    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/inspect?path=corrupt.safetensors"
    )
    check(status == 422 and body["error"]["code"] == "invalid_query",
          "inspect corrupt -> 422 envelope")

    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/folders/via-api", method="PUT"
    )
    check(status == 201 and (ckpt / "via-api").is_dir(), "PUT folder creates it (201)")

    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/files/from-api/x.safetensors",
        method="PUT",
        body_bytes=b"api-bytes",
    )
    check(status == 201 and (ckpt / "from-api" / "x.safetensors").read_bytes() == b"api-bytes",
          "PUT file writes the raw body (201)")

    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/files/../escape.safetensors",
        method="PUT",
        body_bytes=b"x",
    )
    check(status == 422 and body["error"]["code"] == "invalid_query",
          "PUT escape attempt -> 422 envelope")

    status, _, body = asgi_request(
        app, "/api/v1/assets/lora/files/payload.sh",
        method="PUT",
        body_bytes=b"#!/bin/sh",
    )
    check(status == 422 and body["error"]["code"] == "invalid_query",
          "PUT non-safetensors name -> 422 envelope")
    check(not (loras / "payload.sh").exists(), "extension-rejected PUT writes nothing")

    # declared content-length over the cap is refused before the body
    # is read (streaming: a chunked request cannot buffer past the cap)
    status, _, body = asgi_request(
        app, "/api/v1/assets/lora/files/never-written.safetensors",
        method="PUT",
        body_bytes=b"x",
        extra_headers={"content-length": str(64 * 1024 * 1024 * 1024)},
    )
    check(status == 413 and body["error"]["code"] == "asset_too_large",
          "declared body over the cap -> 413 asset_too_large")
    check(not (loras / "never-written.safetensors").exists(),
          "size-rejected PUT writes nothing")

    finish()


if __name__ == "__main__":
    main()
