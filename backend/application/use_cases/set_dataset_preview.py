"""SetDatasetPreview -- point a dataset's card at one item's image.

The client sends an item id, never a path: the preview path is read
from the dataset's own rows, must exist on disk, and is re-checked
for containment -- no path enters this API from the request body.
"""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.dataset_library import DatasetLibrary
from ..ports.dataset_previews import DatasetPreviews


class SetDatasetPreview:
    def __init__(self, *, library: DatasetLibrary, previews: DatasetPreviews) -> None:
        self._library = library
        self._previews = previews

    def execute(self, name: str, item_id: int) -> str:
        # 404 dataset_not_found / dataset_item_not_found and
        # 409 dataset_not_migrated surface from the ports.
        item = self._library.get_item(name, item_id)
        if not item.preview_path:
            raise InvalidQueryError(
                f"item {item_id} in dataset '{name}' has no preview image"
            )
        root = self._library.root(name).resolve()
        target = (root / item.preview_path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise InvalidQueryError(
                f"preview file for item {item_id} is missing on disk"
            )
        self._previews.set(name, item.preview_path)
        return item.preview_path
