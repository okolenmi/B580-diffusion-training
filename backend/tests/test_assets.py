"""Asset domain tests -- real FileSystemAssetStore over temp dirs,
sandboxing behavior, header-only inspect, and the API surface.

Run directly: python backend/tests/test_assets.py
"""

from __future__ import annotations

import tracemalloc

import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import (
    AssetExistsError,
    AssetTooLargeError,
    InvalidQueryError,
)
from backend.application.ports.asset_store import MAX_UPLOAD_BYTES
from backend.application.ports.settings_store import SettingsChanges
from backend.infrastructure.file_asset_store import FileSystemAssetStore
from backend.infrastructure.workspace import WorkspaceDirs, WorkspaceLayout
from backend.presentation.app import create_app
from backend.tests.support import asgi_request, build_services, check, finish


def _layout(root: Path, ckpt: Path, loras: Path) -> WorkspaceLayout:
    """Model directories pinned to this test's own, via WorkspaceDirs.

    Same shape as this helper's earlier version but naming directories
    instead of settings keys: the keys were two strings whose meaning
    ("which directory do checkpoints live in") had to be known
    separately, and a test that set one of them would silently fall
    through to the developer's real ComfyUI for the other.
    """
    return WorkspaceLayout(
        root,
        runs_dir=root / "runs",
        dirs=WorkspaceDirs(checkpoints=ckpt, loras=loras),
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
    store.make_folder("checkpoint", "new/deep")
    check((ckpt / "new" / "deep").is_dir(), "make_folder creates nested folders")

    # Uploads stream: the writer is driven chunk by chunk and never sees
    # the assembled body (docs 08 N-02).
    saved = store.begin_upload("lora", "sub/up.safetensors")
    saved.write(b"payload-")
    saved.write(b"bytes")
    check(Path(saved.finish()).read_bytes() == b"payload-bytes",
          "a chunked write reassembles exactly")

    # The writer is also a context manager, which is what guarantees the
    # abort when the caller raises mid-upload.
    try:
        with store.begin_upload("lora", "boom.safetensors") as w:
            w.write(b"partial")
            raise RuntimeError("simulated failure mid-upload")
    except RuntimeError:
        pass
    check(not (loras / "boom.safetensors").exists(),
          "an aborted upload leaves no final file")
    check(not (loras / "boom.safetensors.part").exists(),
          "and no .part either")

    for bad in ("../evil.safetensors", "/abs.safetensors"):
        try:
            store.begin_upload("lora", bad).write(b"x")
            check(False, f"sandbox rejects upload {bad!r}")
        except InvalidQueryError:
            check(True, f"sandbox rejects upload {bad!r}")

    # upload policy: extension allowlist + size cap, nothing written
    try:
        store.begin_upload("lora", "evil.sh").write(b"#!/bin/sh")
        check(False, "upload rejects a non-safetensors extension")
    except InvalidQueryError:
        check(True, "upload rejects a non-safetensors extension")
    check(not (loras / "evil.sh").exists(), "extension-rejected upload writes nothing")

    # Overwrite protection (N-14): an existing target is refused unless
    # the caller says otherwise. Silent replacement of a real checkpoint
    # is data loss with nothing to tell the user it happened.
    (loras / "keepme.safetensors").write_bytes(b"original")
    try:
        store.begin_upload("lora", "keepme.safetensors").write(b"replacement")
        check(False, "upload refuses an existing target by default")
    except AssetExistsError:
        check(True, "upload refuses an existing target by default")
    check((loras / "keepme.safetensors").read_bytes() == b"original",
          "the refused upload did not touch the existing bytes")
    check(not (loras / "keepme.safetensors.part").exists(),
          "and left no .part behind")

    forced = store.begin_upload("lora", "keepme.safetensors", overwrite=True)
    forced.write(b"replacement")
    forced.finish()
    check((loras / "keepme.safetensors").read_bytes() == b"replacement",
          "overwrite=True replaces it")

    store.max_upload_bytes = 4
    try:
        w = store.begin_upload("lora", "big.safetensors")
        w.write(b"1234")
        w.write(b"5")  # the chunk that crosses the cap
        check(False, "upload enforces the size cap mid-stream")
    except AssetTooLargeError:
        check(True, "upload enforces the size cap mid-stream")
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

    # -- overwrite over HTTP (N-14) ----------------------------------------
    existing = ckpt / "from-api" / "x.safetensors"
    original = existing.read_bytes()
    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/files/from-api/x.safetensors",
        method="PUT", body_bytes=b"replacement",
    )
    check(status == 409 and body["error"]["code"] == "asset_exists",
          f"PUT over an existing file -> 409 asset_exists (got {status})")
    check(existing.read_bytes() == original, "and the original bytes are untouched")

    status, _, body = asgi_request(
        app, "/api/v1/assets/checkpoint/files/from-api/x.safetensors?overwrite=true",
        method="PUT", body_bytes=b"replacement",
    )
    check(status == 201 and existing.read_bytes() == b"replacement",
          f"overwrite=true -> 201 and replaced (got {status})")

    _streaming_upload_checks(app, ckpt)

    finish()


def _streaming_upload_checks(app, ckpt: Path) -> None:
    """The two claims docs 08 N-02 makes about the upload path, as tests.

    Both used to be false, and both were comments rather than tests:
    "cannot buffer the server into swap", and "an ``async def`` route may
    not call blocking file or DB code".

    Memory is measured with tracemalloc rather than RSS, so the number
    is the interpreter's own heap growth and cannot be confused with page
    cache or with this process's other allocations.
    """
    import asyncio

    import httpx

    print("\n== streaming upload: bounded memory, no loop stall (N-02) ==")

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=app)
        # A loopback base_url, because the Host/Origin guard (docs 07
        # F-06) is middleware that these requests really do pass through:
        # an arbitrary host name would be refused with 403 and the test
        # would be measuring the guard instead of the upload.
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://127.0.0.1") as client:

            # (a) 48 MB in 1 MB chunks must not accumulate.
            CHUNK = b"\0" * (1024 * 1024)
            target = ckpt / "streamed.safetensors"
            tracemalloc.start()
            try:
                async def body():
                    for _ in range(48):
                        yield CHUNK

                response = await client.put(
                    "/api/v1/assets/checkpoint/files/streamed.safetensors",
                    content=body(),
                )
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            check(response.status_code == 201, f"48 MB streamed upload -> 201 (got {response.status_code})")
            check(target.stat().st_size == 48 * 1024 * 1024,
                  f"and the whole body landed on disk (got {target.stat().st_size})")
            check(peak < 8 * 1024 * 1024,
                  f"peak heap growth under 8 MB for a 48 MB body "
                  f"(got {peak / 2**20:.1f} MB)")

            # (e) A concurrent GET is served *during* a slow upload.
            # Built from an async generator that awaits between chunks,
            # so the interleaving is the test's own doing rather than a
            # timing threshold: a health check would have to complete
            # between two of our yields.
            saw_health = asyncio.Event()
            health_status: dict[str, object] = {}

            async def slow_body():
                for index in range(8):
                    if index == 2:
                        # Mid-upload: the upload task is parked here, so
                        # anything served now cannot be after it finished.
                        await client.get("/api/v1/health")
                        health_status["status"] = 200
                        saw_health.set()
                    yield b"\0" * (256 * 1024)
                    await asyncio.sleep(0)

            health = asyncio.create_task(
                client.get("/api/v1/health")
            )
            upload = await client.put(
                "/api/v1/assets/checkpoint/files/slow.safetensors",
                content=slow_body(),
            )
            await health
            check(upload.status_code == 201, "slow upload completes")
            check(saw_health.is_set(),
                  "a concurrent health check ran while the upload was open")
            check(health_status.get("status") == 200,
                  "and it was served 200, not queued behind the write")

    asyncio.run(scenario())
    check(not list(ckpt.rglob("*.part")), "no .part left by either streamed upload")


if __name__ == "__main__":
    main()
