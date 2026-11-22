"""Asset catalog and browsing."""

from __future__ import annotations

from pydantic import BaseModel

from ...application.ports.asset_store import AssetCatalog, AssetBrowse


class AssetOptionOut(BaseModel):
    value: str
    label: str


class AssetCatalogOut(BaseModel):
    kind: str
    base_dir: str
    files: list[str]
    options: list[AssetOptionOut]
    upload_supported: bool
    browse_supported: bool


def asset_catalog_out(dto: AssetCatalog) -> AssetCatalogOut:
    return AssetCatalogOut(
        kind=dto.kind,
        base_dir=dto.base_dir,
        files=[option.value for option in dto.options],
        options=[AssetOptionOut(value=o.value, label=o.label) for o in dto.options],
        upload_supported=dto.upload_supported,
        browse_supported=dto.browse_supported,
    )


class AssetBrowseOut(BaseModel):
    kind: str
    path: str
    folders: list[str]
    files: list[str]


def asset_browse_out(dto: AssetBrowse) -> AssetBrowseOut:
    return AssetBrowseOut(
        kind=dto.kind,
        path=dto.path,
        folders=list(dto.folders),
        files=list(dto.files),
    )


class AssetPathOut(BaseModel):
    """Where a write landed (absolute path) for the client that sent
    a relative one."""

    kind: str
    relative_path: str
    path: str


__all__ = ["AssetOptionOut", "AssetCatalogOut", "asset_catalog_out", "AssetBrowseOut", "asset_browse_out", "AssetPathOut"]
