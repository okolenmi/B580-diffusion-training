"""DatasetLibrary port -- managed datasets on disk (format v2).

Storage contract: ``docs/design/backend/04-dataset-format.md``. The
port speaks that format in domain terms; the adapter owns every SQL
statement and every file move.

Reads are the fast path: torch-free own-SQL over each dataset's
``metadata.db``. Two operations deliberately bridge to ``manager``
lazily inside the adapter instead of reimplementing semantics here --
``create`` (schema creation must stay byte-identical to what the
trainer's manager package writes) and ``commit`` (set-membership rules
live there). Those bridges import torch at call time; that is accepted
and documented, not an accident.

Version policy: ``list`` is version-agnostic so the API can *show*
legacy datasets with their ``format_version``; everything else refuses
a non-v2 dataset with ``DatasetNotMigratedError`` (delete excepted --
removing a legacy dataset must stay possible without migrating it).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    """Identity + provenance of one dataset directory."""

    name: str
    description: str
    created_at: datetime
    format_version: int


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Counts over a v2 dataset (never computed for legacy ones)."""

    items: int
    pending: int
    committed: int
    bad: int
    sets: int
    shards: int
    bytes: int


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """One list entry: identity always, stats only for v2 datasets.
    ``preview_path`` is filled by the list use case via the
    ``DatasetPreviews`` port (stored override or first-item fallback);
    the library itself never resolves it."""

    info: DatasetInfo
    stats: DatasetStats | None
    preview_path: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetItem:
    """One trajectory row with its set-membership flag."""

    id: int
    source_id: int
    shard_id: int
    prompt: str
    neg_prompt: str
    model_type: str
    type: str  # 'good' | 'bad'
    cfg: float | None
    seed: int | None
    source_path: str | None
    latent_h: int
    latent_w: int
    preview_path: str | None
    committed: bool


@dataclass(frozen=True, slots=True)
class TrainingSetInfo:
    """A named membership set with its current member count."""

    id: int
    name: str
    description: str | None
    created_at: datetime
    members: int


@dataclass(frozen=True, slots=True)
class ItemChanges:
    """Partial update. ``None`` = leave untouched (an empty string is
    a real value: a cleared caption must be settable as ``""``)."""

    prompt: str | None = None
    neg_prompt: str | None = None
    cfg: float | None = None
    type: str | None = None  # 'good' | 'bad'


@dataclass(frozen=True, slots=True)
class BulkItemChanges:
    """Bulk update, legacy-compatible semantics: ``prompt``/``neg_prompt``
    are truthy-gated (empty string does not clear in bulk), ``cfg`` is
    ``None``-gated, text modes are 'set' | 'prepend' | 'append'
    ('prepend' is the legacy trigger-word flow, idempotent; 'append'
    added M8e for the multi-edit surface), ``type`` is ``None``-gated
    and validated by the use case."""

    prompt: str | None = None
    prompt_mode: str = "set"
    neg_prompt: str | None = None
    neg_prompt_mode: str = "set"
    cfg: float | None = None
    type: str | None = None  # 'good' | 'bad'


class DatasetLibrary(ABC):
    """CRUD + curation over the datasets directory."""

    @abstractmethod
    def list(self) -> tuple[DatasetSummary, ...]:
        """Every dataset directory that contains a ``metadata.db``."""
        raise NotImplementedError

    @abstractmethod
    def get(self, name: str) -> DatasetInfo:
        """One dataset's identity (any format version)."""
        raise NotImplementedError

    @abstractmethod
    def stats(self, name: str) -> DatasetStats:
        """Counts for a v2 dataset (``DatasetNotMigratedError`` on v1)."""
        raise NotImplementedError

    @abstractmethod
    def root(self, name: str) -> Path:
        """Validated absolute directory of a dataset that exists.

        Name validation happens here (no separators, no traversal):
        the returned path is the only handle child processes get.
        """
        raise NotImplementedError

    @abstractmethod
    def create(self, name: str, description: str | None = None) -> DatasetInfo:
        """Create a fresh v2 dataset (schema via the manager bridge)."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Remove the dataset directory. Allowed for any format version.
        Returns False when it did not exist."""
        raise NotImplementedError

    @abstractmethod
    def list_items(
        self, name: str, *, committed: bool | None = None
    ) -> tuple[DatasetItem, ...]:
        """All items, or filtered by training-set membership."""
        raise NotImplementedError

    @abstractmethod
    def get_item(self, name: str, item_id: int) -> DatasetItem:
        """One trajectory row (``DatasetItemNotFoundError`` when absent,
        ``DatasetNotMigratedError`` on a legacy dataset)."""
        raise NotImplementedError

    @abstractmethod
    def first_preview(self, name: str) -> str | None:
        """First non-bad item's ``preview_path`` (id order).

        Best-effort display fallback for the dataset card: ``None``
        for a missing dataset, a legacy (pre-v2) one, or a dataset
        without preview files -- never raises."""
        raise NotImplementedError

    @abstractmethod
    def update_item(self, name: str, item_id: int, changes: ItemChanges) -> DatasetItem:
        """Apply a partial update to one item."""
        raise NotImplementedError

    @abstractmethod
    def bulk_update(
        self, name: str, item_ids: list[int], changes: BulkItemChanges
    ) -> int:
        """Bulk update; returns the number of rows matched."""
        raise NotImplementedError

    @abstractmethod
    def discard(self, name: str, item_ids: list[int]) -> int:
        """Delete items (membership cascades), their preview files, and
        any shard left with no rows; returns rows removed."""
        raise NotImplementedError

    @abstractmethod
    def list_sets(self, name: str) -> tuple[TrainingSetInfo, ...]:
        """Training sets with member counts, newest first."""
        raise NotImplementedError

    @abstractmethod
    def commit(self, name: str, item_ids: list[int], set_name: str) -> int:
        """Add items to a named training set (creating/reusing it);
        returns the set id. Membership only -- files never move."""
        raise NotImplementedError
