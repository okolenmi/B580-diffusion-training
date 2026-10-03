"""DatasetFiles adapter -- scoped reads of dataset preview images.

Two rules, both enforced here where the disk actually is:

* **Containment.** Nothing outside ``datasets/{name}/`` is ever
  reachable. Traversal in `rel_path` is refused rather than resolved,
  and so is a symlink pointing out of the tree.
* **Allowlist.** Only preview *images* are served: ``.png``, ``.jpg``,
  ``.jpeg``, ``.webp`` (case-insensitive), within
  :data:`MAX_PREVIEW_BYTES`.

The second rule exists because this port's only caller is the items
grid's ``<img>`` (frontend/js/views/datasets.js's ``previewUrl``). It
used to serve *any* file under the dataset directory, which meant
``metadata.db`` and multi-GB ``.safetensors`` shards came back over the
same route -- the latter read fully into memory by ``read_bytes()`` --
and ``.svg`` came back as ``image/svg+xml``, which executes script on
this app's own origin. That last one is not a theoretical concern in a
same-origin app whose Origin check trusts its own Host: script that
origin would pass the check.

So the allowlist is not a performance measure, it is what the route
claims to be. A non-image is reported as not-found rather than as
"forbidden": from this route's point of view there is no such preview.
"""

from __future__ import annotations

from pathlib import Path

from ..application.errors import DatasetFileNotFoundError, DatasetNotFoundError
from ..application.limits import MAX_PREVIEW_BYTES
from ..application.ports.dataset_files import DatasetFile, DatasetFiles

#: The preview extensions this route serves. Anything else is a 404.
PREVIEW_SUFFIXES: dict[str, str] = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

#: Largest preview served, checked with ``stat()`` *before* the read, so
#: an oversized file is never pulled into memory just to be rejected.


class FsDatasetFiles(DatasetFiles):
    def __init__(self, datasets_dir: Path) -> None:
        self._datasets_dir = datasets_dir

    def read(self, dataset: str, rel_path: str) -> DatasetFile:
        # A NUL byte is not a path, it is the string's own terminator, and
        # every filesystem call below raises ValueError on one rather than
        # returning "no such file". Both halves of the URL reach the disk
        # here, so both are checked before anything is resolved -- an
        # unhandled ValueError from lstat() reaches the client as a 500,
        # and a name that cannot exist deserves the same not-found answer
        # as one that does not.
        if "\x00" in dataset:
            raise DatasetNotFoundError(f"dataset '{dataset}' not found")
        if "\x00" in rel_path:
            raise DatasetFileNotFoundError(
                f"file '{rel_path}' does not exist in dataset '{dataset}'"
            )

        root = (self._datasets_dir / dataset).resolve()
        if not root.is_relative_to(self._datasets_dir.resolve()):
            # name itself escapes (e.g. ".."): not a dataset of ours
            raise DatasetNotFoundError(f"dataset '{dataset}' not found")
        if not root.is_dir():
            raise DatasetNotFoundError(f"dataset '{dataset}' not found")

        target = (root / rel_path).resolve()
        if not target.is_relative_to(root):
            raise DatasetFileNotFoundError(
                f"file '{rel_path}' is outside dataset '{dataset}'"
            )

        # Allowlist by suffix, before the filesystem is consulted further:
        # this decides what the route *is*, not whether a path happens to
        # resolve. Case-insensitive, and checked on the resolved name so
        # "x.PNG" and "x.png" behave the same.
        media_type = PREVIEW_SUFFIXES.get(target.suffix.lower())
        if media_type is None:
            raise DatasetFileNotFoundError(
                f"'{rel_path}' is not a preview image "
                f"(allowed: {', '.join(sorted(PREVIEW_SUFFIXES))})"
            )

        # ``is_file()`` after resolve() already excludes a symlink to a
        # directory; a symlink to a *file* outside the dataset was
        # refused above, because resolve() followed it.
        if not target.is_file():
            raise DatasetFileNotFoundError(
                f"file '{rel_path}' not found in dataset '{dataset}'"
            )

        # Size from the stat, not from len(bytes): the point is to refuse
        # before reading, not to read and then notice.
        try:
            size = target.stat().st_size
        except OSError as exc:
            raise DatasetFileNotFoundError(
                f"file '{rel_path}' is not readable: {exc}"
            ) from exc
        if size > MAX_PREVIEW_BYTES:
            raise DatasetFileNotFoundError(
                f"preview '{rel_path}' is {size} bytes, over the "
                f"{MAX_PREVIEW_BYTES}-byte limit"
            )

        return DatasetFile(content=target.read_bytes(), media_type=media_type)