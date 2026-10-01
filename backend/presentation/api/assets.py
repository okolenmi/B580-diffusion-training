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
    responses={413: {"description": "body over MAX_UPLOAD_BYTES"}, 422: _ERROR_422},
)
async def upload_asset(
    kind: str,
    relative_path: str,
    request: Request,
    services: ApplicationServices = Depends(get_services),
) -> AssetPathOut:
    """Raw-body upload: the request body *is* the file.

    Bounded work: a declared Content-Length above the cap is rejected
    before the body is read, and the read itself stops at the cap, so
    neither a honest-large nor a chunked-lying request can buffer the
    server into swap (the adapter re-checks while writing).
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise AssetTooLargeError(
            f"upload of {relative_path!r} declares {declared} bytes; "
            f"the cap is {MAX_UPLOAD_BYTES} bytes"
        )
    total = 0
    chunks: list[bytes] = []
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise AssetTooLargeError(
                f"upload of {relative_path!r} exceeds the "
                f"{MAX_UPLOAD_BYTES}-byte cap"
            )
        chunks.append(chunk)
    saved = services.assets.upload.execute(kind, relative_path, b"".join(chunks))
    return AssetPathOut(kind=kind, relative_path=relative_path, path=saved)
