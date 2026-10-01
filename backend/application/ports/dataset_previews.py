"""DatasetPreviews -- which item image represents a dataset (M8f).

Server-side view state stored in backend.db, following the same rule
as task rows: the pointer lives with the server, never inside a
dataset's metadata.db -- a dataset folder stays exactly what the
trainer wrote (format v2 removed dataset-side server state; see
``docs/design/backend/04-dataset-format.md``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DatasetPreviews(ABC):
    """Effective card preview per dataset."""

    @abstractmethod
    def resolve(self, name: str) -> str | None:
        """Effective dataset-relative preview path.

        The stored override wins when its file still exists (a
        discarded item must not leave a dead image on the card), else
        the first non-bad item's preview, else ``None``. Best-effort by
        contract: a missing or legacy dataset resolves to ``None``,
        never raises.
        """

    @abstractmethod
    def set(self, name: str, preview_path: str) -> None:
        """Store the override (the use case validates it first)."""

    @abstractmethod
    def remove(self, name: str) -> None:
        """Drop the stored override. Idempotent -- called when the
        dataset is deleted so a later dataset with the same name can
        never inherit a stranger's preview pointer."""
