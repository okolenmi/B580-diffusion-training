"""Assets endpoints -- list/browse/inspect model files, write into them.

Writes are plain ``PUT``s of the target resource (no multipart):

* ``PUT /{kind}/files/{path}`` -- body is the file's bytes;
* ``PUT /{kind}/folders/{path}`` -- create the folder (idempotent).

Client paths are sandboxed server-side against the kind's base
directory; any escape attempt or unknown kind is ``invalid_query``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from starlette.concurrency import run_in_threadpool

from ...application.errors import AssetTooLargeError
from ...application.ports.asset_store import MAX_UPLOAD_BYTES
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import (
    AssetBrowseOut,
    AssetCatalogOut,
    AssetPathOut,
    asset_browse_out,
    asset_catalog_out,
)

router = APIRouter(prefix="/api/v1/assets", tags=["assets"])

_ERROR_422 = {"description": "unknown kind, invalid path, or sandbox violation"}


@router.get("/{kind}", response_model=AssetCatalogOut, responses={422: _ERROR_422})
def list_assets(
    kind: str,
    services: ApplicationServices = Depends(get_services),
) -> AssetCatalogOut:
    """One round-trip for a file picker: base dir, entries, capabilities."""
    return asset_catalog_out(services.assets.list.execute(kind))


@router.get("/{kind}/browse", response_model=AssetBrowseOut, responses={422: _ERROR_422})
def browse_assets(
    kind: str,
    path: str = Query(""),
    services: ApplicationServices = Depends(get_services),
) -> AssetBrowseOut:
    """Immediate children of one directory (root when ``path`` is empty)."""
    return asset_browse_out(services.assets.browse.execute(kind, path))


@router.get("/{kind}/inspect", response_model=dict[str, Any], responses={422: _ERROR_422})
def inspect_asset(
    kind: str,
    path: str = Query(""),
    services: ApplicationServices = Depends(get_services),
) -> dict[str, Any]:
    """Header-only safetensors metadata. Fixed contract per kind:
    checkpoints yield ``{kind, path, components}``, LoRAs yield
    ``{kind, path, dtype, rank, key_count}``."""
    return services.assets.inspect.execute(kind, path)


@router.put(
    "/{kind}/folders/{relative_path:path}",
    response_model=AssetPathOut,
    status_code=201,
    responses={422: _ERROR_422},
)
def make_asset_folder(
    kind: str,
    relative_path: str,
    services: ApplicationServices = Depends(get_services),
) -> AssetPathOut:
    """Create (idempotently) and report the absolute path."""
    created = services.assets.make_folder.execute(kind, relative_path)
    return AssetPathOut(kind=kind, relative_path=relative_path, path=created)


@router.put(
    "/{kind}/files/{relative_path:path}",
    response_model=AssetPathOut,
    status_code=201,
    responses={
        409: {"description": "target exists and overwrite was not requested"},
        413: {"description": "body over MAX_UPLOAD_BYTES"},
        422: _ERROR_422,
    },
)
async def upload_asset(
    kind: str,
    relative_path: str,
    request: Request,
    overwrite: bool = Query(
        False,
        description="replace the target if it exists. Off by default: a PUT "
                    "that silently overwrites a real checkpoint is data "
                    "loss (docs 08 N-14).",
    ),
    services: ApplicationServices = Depends(get_services),
) -> AssetPathOut:
    """Raw-body upload: the request body *is* the file.

    Streamed, never buffered (docs 08 N-02). The previous version
    collected every chunk into a list, joined them, and wrote the result
    on the event loop; measured on a real server, a 600 MB upload took
    the loop from 9 ms to 971 ms of latency for concurrent requests and
    held about 2x the file in RAM. With an 8 GiB cap that is not a
    bounded cost, and the docstring's old claim that neither request
    style could "buffer the server into swap" was simply false.

    So: one chunk in memory at a time, and every write goes through
    `anyio.to_thread.run_sync`, because `write`/`fsync`/`replace` block
    and an `async def` route must not block the loop it runs on (quality
    rule 5). The size cap is checked twice, and neither check trusts the
    request: once against a declared Content-Length (a cheap early exit)
    and once inside the writer against the running total (the real one,
    since a chunked request has no length to lie about or to be honest
    about).
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise AssetTooLargeError(
            f"upload of {relative_path!r} declares {declared} bytes; "
            f"the cap is {MAX_UPLOAD_BYTES} bytes"
        )

    # Reads happen here (on the loop); writes happen off it. The session
    # is where every rule was already applied, so this only moves bytes.
    session = services.assets.upload.begin(kind, relative_path, overwrite=overwrite)
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                # Second, authoritative check: a chunked request has no
                # length to check up front, so the running total is the
                # only place the cap can actually be enforced. The
                # writer enforces it too -- this one is here so the
                # client is told before we write the offending chunk.
                raise AssetTooLargeError(
                    f"upload of {relative_path!r} exceeds the "
                    f"{MAX_UPLOAD_BYTES}-byte cap"
                )
            await run_in_threadpool(session.write, chunk)
        saved = await run_in_threadpool(session.finish)
    except BaseException:
        # Includes cancellation: a client that disconnects mid-upload
        # must not leave a .part behind for the next run to append to.
        session.abort()
        raise
    return AssetPathOut(kind=kind, relative_path=relative_path, path=saved)
