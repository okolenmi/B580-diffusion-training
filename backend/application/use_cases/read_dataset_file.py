"""ReadDatasetFile -- one scoped file read from a dataset directory.

Thin on purpose: every rule that touches the filesystem (existence,
containment) is enforced by the port's adapter, where the disk
actually is. The use case exists so the route has a single entry
point like every other one, and so an empty path is rejected before
it reaches the disk.
"""

from __future__ import annotations

from ..errors import InvalidQueryError
from ..ports.dataset_files import DatasetFile, DatasetFiles


class ReadDatasetFile:
    def __init__(self, *, files: DatasetFiles) -> None:
        self._files = files

    def execute(self, dataset: str, rel_path: str) -> DatasetFile:
        if not dataset:
            raise InvalidQueryError("dataset name is required")
        if not rel_path or rel_path.endswith("/"):
            raise InvalidQueryError("file path is required")
        return self._files.read(dataset, rel_path)
