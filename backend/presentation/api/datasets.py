"""Dataset endpoints -- library, curation, training sets, tasks (M3b).

Resource-oriented contract over ``services.datasets``:

* ``GET/POST /api/v1/datasets`` -- list summaries / create;
* ``GET/DELETE /{name}`` -- detail (v1 datasets: 409
  ``dataset_not_migrated``) / remove (allowed at any format version);
* ``GET /{name}/items`` -- rows (``committed=true|false`` membership
  filter); ``PATCH /{name}/items`` -- bulk edit; ``PATCH
  /{name}/items/{id}`` -- single edit (explicit ``type`` replaces the
  legacy toggle endpoint); ``POST /{name}/items/discard``;
* ``GET /{name}/files/{path}`` -- preview image bytes (containment
  enforced by the adapter, never resolved out of the dataset dir);
  ``PUT /{name}/preview`` -- front the dataset's card with one item's
  preview image (body ``{item_id}``);
* ``GET/POST /{name}/sets`` -- list / commit items into a set;
* ``GET/POST /{name}/tasks`` -- rows / start; ``POST
  /{name}/tasks/{id}/stop`` -- SIGKILL.

Every error leaves as the one envelope (``dataset_not_found`` 404,
``dataset_task_active`` 409, ...); see ``presentation/errors.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Response

from ...application.limits import DEFAULT_DATASET_ITEM_PAGE_SIZE, MAX_PAGE_SIZE
from ...application.services import ApplicationServices
from ..deps import get_services
from ..schemas import (
    BulkUpdateItemsIn,
    BulkUpdateOut,
    CommitItemsIn,
    CommitOut,
    CreateDatasetIn,
    DatasetDeletedOut,
    DatasetDetailOut,
    DatasetInfoOut,
    DatasetItemOut,
    DatasetItemsOut,
    DatasetListOut,
    DatasetPreviewOut,
    DatasetSetsOut,
    DatasetTaskOut,
    DatasetTasksOut,
    DiscardOut,
    ItemIdsIn,
    SetPreviewIn,
    StartDatasetTaskIn,
    UpdateItemIn,
    dataset_detail_out,
    dataset_info_out,
    dataset_item_out,
    dataset_list_out,
    dataset_sets_out,
    dataset_task_out,
    dataset_tasks_out,
)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

_ERROR_404 = {"description": "dataset or item not found"}
_ERROR_409 = {"description": "task already active / not migrated / already exists"}

# The ceiling the use case enforces, imported rather than repeated, so
# the OpenAPI doc and the 422 can never disagree (docs 08 S-08).
MAX_ITEM_PAGE = MAX_PAGE_SIZE
# The route's documented default and the use case's applied default
# are the same constant, so a 422 and an OpenAPI description cannot
# disagree about what "no limit" means.
DEFAULT_DATASET_ITEM_PAGE = DEFAULT_DATASET_ITEM_PAGE_SIZE


@router.get("", response_model=DatasetListOut)
def list_datasets(services: ApplicationServices = Depends(get_services)):
    """Every dataset with identity; ``stats`` is null until migrated."""
    return dataset_list_out(services.datasets.list.execute())


@router.post(
    "", response_model=DatasetInfoOut, status_code=201, responses={409: _ERROR_409}
)
def create_dataset(
    body: CreateDatasetIn,
    services: ApplicationServices = Depends(get_services),
):
    """Fresh v2 dataset directory (schema via the manager bridge)."""
    return dataset_info_out(services.datasets.create.execute(body.name, body.description))


@router.get(
    "/{name}", response_model=DatasetDetailOut, responses={404: _ERROR_404, 409: _ERROR_409}
)
def get_dataset(name: str, services: ApplicationServices = Depends(get_services)):
    """One round-trip: identity, counts, sets, active tasks."""
    return dataset_detail_out(services.datasets.get.execute(name))


@router.delete(
    "/{name}", response_model=DatasetDeletedOut, responses={404: _ERROR_404, 409: _ERROR_409}
)
def delete_dataset(name: str, services: ApplicationServices = Depends(get_services)):
    """Remove the dataset; refuses while a task is active."""
    deleted = services.datasets.delete.execute(name)
    return DatasetDeletedOut(deleted=deleted)


@router.get("/{name}/items", response_model=DatasetItemsOut, responses={404: _ERROR_404, 409: _ERROR_409})
def list_dataset_items(
    name: str,
    committed: bool | None = Query(None),
    limit: int | None = Query(
        None,
        ge=1,
        le=MAX_ITEM_PAGE,
        description=f"rows per page (default: {DEFAULT_DATASET_ITEM_PAGE_SIZE})",
    ),
    offset: int = Query(0, ge=0),
    services: ApplicationServices = Depends(get_services),
):
    """One page of trajectory rows, optionally by training-set membership.

    Paged by default (docs 08 Q10). The response carries `total` and
    `next_offset` so a client can tell a truncated page from the whole
    set and ask for the rest; `next_offset` is null on the last page.
    """
    result = services.datasets.items.execute(
        name, committed=committed, limit=limit, offset=offset
    )
    return DatasetItemsOut(
        items=[dataset_item_out(item) for item in result.items],
        count=result.count,
        limit=result.limit,
        offset=result.offset,
        total=result.total,
        next_offset=result.next_offset,
    )


@router.patch(
    "/{name}/items", response_model=BulkUpdateOut, responses={404: _ERROR_404, 409: _ERROR_409}
)
def bulk_update_items(
    name: str,
    body: BulkUpdateItemsIn,
    services: ApplicationServices = Depends(get_services),
):
    """Multi-row edit: text fields with set/prepend/append, CFG, verdict
    (legacy truthy semantics -- see the use case)."""
    result = services.datasets.bulk_update.execute(
        name,
        list(body.item_ids),
        prompt=body.prompt,
        prompt_mode=body.prompt_mode,
        neg_prompt=body.neg_prompt,
        neg_prompt_mode=body.neg_prompt_mode,
        cfg=body.cfg,
        type=body.type,
    )
    return BulkUpdateOut(updated=result.updated)


@router.patch(
    "/{name}/items/{item_id}",
    response_model=DatasetItemOut,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def update_dataset_item(
    name: str,
    item_id: int,
    body: UpdateItemIn,
    services: ApplicationServices = Depends(get_services),
):
    """Single-row partial edit; returns the refreshed item."""
    item = services.datasets.update_item.execute(
        name,
        item_id,
        prompt=body.prompt,
        neg_prompt=body.neg_prompt,
        cfg=body.cfg,
        type=body.type,
    )
    return dataset_item_out(item)


@router.post(
    "/{name}/items/discard",
    response_model=DiscardOut,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def discard_items(
    name: str,
    body: ItemIdsIn,
    services: ApplicationServices = Depends(get_services),
):
    """Delete rows (+ previews, empty shards); membership cascades."""
    result = services.datasets.discard.execute(name, list(body.item_ids))
    return DiscardOut(deleted=result.deleted)


@router.get(
    "/{name}/files/{rel_path:path}",
    response_class=Response,
    responses={404: _ERROR_404},
)
def read_dataset_file(
    name: str,
    rel_path: str,
    services: ApplicationServices = Depends(get_services),
):
    """Preview image bytes for the items grid.

**Allowlist: ``.png``, ``.jpg``, ``.jpeg``, ``.webp`` only**, at most
32 MB, case-insensitive. Anything else is
``dataset_file_not_found`` (404) -- including files that exist. The
adapter enforces the list and the cap, because that is where the disk
is; see `infrastructure/dataset_files.py`.

Scoped by contract too: the adapter refuses anything outside
``datasets/{name}/`` (``dataset_file_not_found`` 404), so this route can
never be steered at the rest of the filesystem.

This route is *only* for the grid's `<img>` (the UI's `previewUrl`).
It is deliberately not a general file endpoint for a dataset directory:
it used to be one, and served ``metadata.db``, whole ``.safetensors``
shards read into memory, and ``.svg`` as active content on this app's
own origin (docs 08 N-05).
"""
    payload = services.datasets.read_file.execute(name, rel_path)
    # Headers are not decoration here. This route serves user-supplied
    # bytes on the app's own origin, so:
    #   nosniff          -- stop the browser from sniffing a type we did
    #                       not declare (and overriding ours anyway);
    #   CSP sandbox      -- even if a file did get served as active
    #                       content, `sandbox` denies it script, same-
    #                       origin access and form submission;
    #   Content-Disposition attachment -- an image the browser renders
    #                       inline is fine, but this also prevents any
    #                       future non-image from being interpreted as
    #                       a document in this origin;
    #   private, short max-age -- previews are per-dataset user data,
    #                       not shared-cacheable, and they change when
    #                       the item is re-previewed.
    # The allowlist itself (png/jpg/jpeg/webp, size-capped) is enforced
    # by the port's adapter, so a .db or a .safetensors shard is a 404
    # rather than something these headers have to defend against
    # (docs 08 N-05).
    return Response(
        content=payload.content,
        media_type=payload.media_type,
        headers={
            "Cache-Control": "private, max-age=60",
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
            "Content-Disposition": f'inline; filename="{name}-preview"',
        },
    )


@router.put(
    "/{name}/preview",
    response_model=DatasetPreviewOut,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def set_dataset_preview(
    name: str,
    body: SetPreviewIn,
    services: ApplicationServices = Depends(get_services),
):
    """Front the dataset's card with this item's preview image.

    404 unknown dataset/item, 409 legacy dataset, 422 ``invalid_query``
    when the item has no preview or its file is gone.
    """
    return DatasetPreviewOut(
        preview_path=services.datasets.set_preview.execute(name, body.item_id)
    )


@router.get("/{name}/sets", response_model=DatasetSetsOut, responses={404: _ERROR_404, 409: _ERROR_409})
def list_dataset_sets(name: str, services: ApplicationServices = Depends(get_services)):
    return dataset_sets_out(services.datasets.sets.execute(name))


@router.post(
    "/{name}/sets",
    response_model=CommitOut,
    status_code=201,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def commit_items(
    name: str,
    body: CommitItemsIn,
    services: ApplicationServices = Depends(get_services),
):
    """Membership-only commit into the named set (reused by name)."""
    result = services.datasets.commit.execute(
        name, list(body.item_ids), set_name=body.name
    )
    return CommitOut(set_id=result.set_id, set_name=result.set_name, added=result.added)


@router.get("/{name}/tasks", response_model=DatasetTasksOut, responses={404: _ERROR_404, 409: _ERROR_409})
def list_dataset_tasks(
    name: str,
    active_only: bool = Query(True),
    services: ApplicationServices = Depends(get_services),
):
    """Newest first; sweeps rows whose child died unreported."""
    return dataset_tasks_out(services.datasets.tasks.execute(name, active_only=active_only))


@router.post(
    "/{name}/tasks",
    response_model=DatasetTaskOut,
    status_code=201,
    responses={409: _ERROR_409},
)
def start_dataset_task(
    name: str,
    body: StartDatasetTaskIn,
    services: ApplicationServices = Depends(get_services),
):
    """Spawn one ingestion child (single active task per dataset)."""
    return dataset_task_out(services.datasets.start_task.execute(body.to_command(name)))


@router.post(
    "/{name}/tasks/{task_id}/stop",
    response_model=DatasetTaskOut,
    responses={404: _ERROR_404, 409: _ERROR_409},
)
def stop_dataset_task(
    name: str, task_id: int, services: ApplicationServices = Depends(get_services)
):
    """SIGKILL the task group; CASed so a finished task stays finished."""
    return dataset_task_out(services.datasets.stop_task.execute(task_id))
