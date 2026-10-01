"""SqliteDatasetLibrary -- own-SQL reads over dataset metadata.db files.

Every read below is torch-free: direct SQL over the v2 columns of
``docs/design/backend/04-dataset-format.md`` (no ``json_extract``, no
``manager`` import). Two methods bridge ``manager`` *lazily inside the
method body* -- ``create`` (schema creation must stay byte-identical to
the trainer's package) and ``commit`` (membership semantics live
there). Those two pull torch in at call time; documented trade-off.

Connections mirror ``manager.db._connect``'s pragmas (WAL +
busy_timeout + foreign_keys) so the server, a trainer child, and an
ingestion child can share the file safely.

Version policy (see the port): ``list``/``get`` work on any version,
``delete`` works on any version, everything else refuses a non-v2
dataset with ``DatasetNotMigratedError`` before touching v2 columns.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..application.errors import (
    DatasetAlreadyExistsError,
    DatasetDirectoryConflictError,
    DatasetItemNotFoundError,
    DatasetNotFoundError,
    DatasetNotMigratedError,
    InvalidQueryError,
)
from ..application.ports.dataset_library import (
    BulkItemChanges,
    DatasetInfo,
    DatasetItem,
    DatasetLibrary,
    DatasetStats,
    DatasetSummary,
    ItemChanges,
    TrainingSetInfo,
)
from .workspace import WorkspaceLayout

logger = logging.getLogger(__name__)


def _dt(epoch: float | None) -> datetime:
    if not epoch:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromtimestamp(float(epoch), tz=timezone.utc)


class SqliteDatasetLibrary(DatasetLibrary):
    def __init__(self, layout: WorkspaceLayout) -> None:
        self._layout = layout

    # -- listing / identity ----------------------------------------------

    def list(self) -> tuple[DatasetSummary, ...]:
        base = self._layout.datasets_dir
        if not base.is_dir():
            return ()
        summaries: list[DatasetSummary] = []
        for child in sorted(base.iterdir()):
            if child.name.startswith(".") or not child.is_dir():
                continue
            if not (child / "metadata.db").exists():
                continue  # ghost or unrelated directory
            info = self._read_info(child)
            version = self._version(child / "metadata.db")
            stats = self._stats(child) if version == 2 else None
            summaries.append(DatasetSummary(info=info, stats=stats))
        return tuple(summaries)

    def get(self, name: str) -> DatasetInfo:
        return self._read_info(self._existing_dir(name))

    def root(self, name: str) -> Path:
        return self._existing_dir(name)

    def stats(self, name: str) -> DatasetStats:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        return self._stats(directory)

    # -- lifecycle --------------------------------------------------------

    def create(self, name: str, description: str | None = None) -> DatasetInfo:
        directory = self._validate_dir(name)
        if directory.exists():
            if (directory / "metadata.db").exists():
                raise DatasetAlreadyExistsError(f"dataset '{name}' already exists")
            # No metadata.db, so nothing in it can be loadable -- but a
            # directory with content is not ours to delete: it may be the
            # user's own image folder that happens to share the name.
            # Only an empty skeleton (what a create() interrupted before
            # writing the database leaves behind) is cleared
            # (docs 07 F-10).
            self._clear_empty_skeleton(name, directory)
        directory.mkdir(parents=True)
        (directory / "shards").mkdir()
        (directory / "previews").mkdir()

        # Documented adapter bridge: schema must come from the same
        # code the trainer writes with, or the two can drift.
        from manager.db import init_local_db, set_dataset_info  # noqa: PLC0415

        init_local_db(directory / "metadata.db")
        set_dataset_info(directory / "metadata.db", name, description)
        return self._read_info(directory)

    @staticmethod
    def _clear_empty_skeleton(name: str, directory: Path) -> None:
        """Remove ``directory`` when it holds nothing but the empty
        skeleton ``create`` makes; otherwise refuse the name."""
        strays = [
            child
            for child in directory.iterdir()
            if not (
                child.is_dir()
                and child.name in {"shards", "previews"}
                and not any(child.iterdir())
            )
        ]
        if strays:
            names = ", ".join(sorted(child.name for child in strays)[:5])
            raise DatasetDirectoryConflictError(
                f"'{directory}' already exists and is not a dataset "
                f"(found: {names}). Rename or remove it, then create the "
                f"dataset '{name}' -- this server never deletes files it "
                f"did not create."
            )
        shutil.rmtree(directory)

    def delete(self, name: str) -> bool:
        directory = self._validate_dir(name)
        if not directory.exists():
            return False
        if not (directory / "metadata.db").exists():
            raise DatasetNotFoundError(f"no dataset named '{name}'")
        shutil.rmtree(directory)
        return True

    # -- items ------------------------------------------------------------

    def list_items(
        self,
        name: str,
        *,
        committed: bool | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[DatasetItem, ...]:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        sql = (
            "SELECT t.*, (EXISTS (SELECT 1 FROM set_members sm "
            "WHERE sm.trajectory_id = t.id)) AS committed "
            "FROM trajectories t"
        )
        params: tuple = ()
        if committed is True:
            sql += " WHERE committed"
        elif committed is False:
            sql += " WHERE NOT committed"
        sql += " ORDER BY t.id"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params = (limit, offset)
        elif offset:
            sql += " LIMIT -1 OFFSET ?"
            params = (offset,)
        with self._connect(directory / "metadata.db") as conn:
            rows = conn.execute(sql, params).fetchall()
        return tuple(_row_to_item(r) for r in rows)

    def get_item(self, name: str, item_id: int) -> DatasetItem:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        item = self._get_item(directory, item_id)
        if item is None:
            raise DatasetItemNotFoundError(
                f"no trajectory {item_id} in dataset '{name}'"
            )
        return item

    def first_preview(self, name: str) -> str | None:
        # Best-effort display fallback (port contract): a missing or
        # legacy dataset resolves to None instead of raising, so a
        # list round-trip never dies on one odd directory.
        directory = self._validate_dir(name)
        db = directory / "metadata.db"
        if not db.exists() or self._version(db) != 2:
            return None
        with self._connect(db) as conn:
            row = conn.execute(
                "SELECT preview_path FROM trajectories "
                "WHERE preview_path IS NOT NULL AND preview_path != '' "
                "AND type != 'bad' "
                "ORDER BY id LIMIT 1"
            ).fetchone()
        return str(row[0]) if row else None

    def update_item(self, name: str, item_id: int, changes: ItemChanges) -> DatasetItem:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        sets: list[str] = []
        params: list = []
        for column, value in (
            ("prompt", changes.prompt),
            ("neg_prompt", changes.neg_prompt),
            ("cfg", changes.cfg),
            ("type", changes.type),
        ):
            if value is None:
                continue
            sets.append(f"{column} = ?")
            params.append(value)
        with self._connect(directory / "metadata.db") as conn:
            existing = conn.execute(
                "SELECT id FROM trajectories WHERE id = ?", (item_id,)
            ).fetchone()
            if existing is None:
                raise DatasetItemNotFoundError(
                    f"no trajectory {item_id} in dataset '{name}'"
                )
            conn.execute(
                f"UPDATE trajectories SET {', '.join(sets)} WHERE id = ?",
                (*params, item_id),
            )
        refreshed = self._get_item(directory, item_id)
        assert refreshed is not None
        return refreshed

    def bulk_update(
        self, name: str, item_ids: list[int], changes: BulkItemChanges
    ) -> int:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        placeholders = ",".join("?" * len(item_ids))
        with self._connect(directory / "metadata.db") as conn:
            rows = conn.execute(
                f"SELECT id, prompt, neg_prompt, cfg, type FROM trajectories "
                f"WHERE id IN ({placeholders})",
                item_ids,
            ).fetchall()
            updated = 0
            for row in rows:
                prompt = self._merge_text(
                    row["prompt"] or "", changes.prompt, changes.prompt_mode
                )
                neg_prompt = self._merge_text(
                    row["neg_prompt"] or "",
                    changes.neg_prompt,
                    changes.neg_prompt_mode,
                )
                # Truthy gates are legacy-compatible: an empty string
                # never clears in bulk (single-item PATCH does that).
                conn.execute(
                    "UPDATE trajectories SET prompt = ?, neg_prompt = ?, "
                    "cfg = ?, type = ? WHERE id = ?",
                    (
                        prompt,
                        neg_prompt,
                        changes.cfg if changes.cfg is not None else row["cfg"],
                        changes.type if changes.type is not None else row["type"],
                        row["id"],
                    ),
                )
                updated += 1
        return updated

    @staticmethod
    def _merge_text(existing: str, new: str | None, mode: str) -> str:
        """set / prepend / append, both merges idempotent (re-applying
        the same value is a no-op instead of doubling the caption).

        The join adds a separating space only when the incoming text
        brings no separator of its own: tags are comma-separated, so
        appending ", ugly" must read "lowres, ugly", never
        "lowres , ugly" -- while "trigger" still lands as "photo 2
        trigger" (the legacy prepend shape)."""

        def join(first: str, second: str) -> str:
            if not first:
                return second
            if not second:
                return first
            if first[-1].isspace() or second[0].isspace() or second[0] == ",":
                return f"{first}{second}"
            return f"{first} {second}"

        if not new:
            return existing  # truthy gate, legacy-compatible
        if mode == "set":
            return new
        if mode == "prepend":
            if existing.startswith(new):
                return existing
            return join(new, existing)
        # "append"
        if existing.endswith(new):
            return existing
        return join(existing, new)

    def discard(self, name: str, item_ids: list[int]) -> int:
        """Delete rows, then remove the files they owned.

        Ordering is the whole point (docs 07 F-05): the rows go inside
        one transaction, the unlink happens *after* the commit. Deleting
        a file first and then rolling back leaves the dataset referencing
        a shard that no longer exists -- dangling rows are
        unrecoverable, an orphaned file is not. Removal failures are
        logged and survive as orphans rather than undoing the discard.
        """
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        placeholders = ",".join("?" * len(item_ids))
        db = directory / "metadata.db"
        orphaned: list[Path] = []
        with self._connect(db) as conn:
            rows = conn.execute(
                f"SELECT preview_path, shard_id FROM trajectories "
                f"WHERE id IN ({placeholders})",
                item_ids,
            ).fetchall()
            deleted = conn.execute(
                f"DELETE FROM trajectories WHERE id IN ({placeholders})", item_ids
            ).rowcount
            # set_members rows go with the FK cascade (foreign_keys=ON).
            shard_ids = {r["shard_id"] for r in rows}
            for shard_id in shard_ids:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM trajectories WHERE shard_id = ?",
                    (shard_id,),
                ).fetchone()[0]
                if remaining:
                    continue
                shard = conn.execute(
                    "SELECT file_path FROM shards WHERE id = ?", (shard_id,)
                ).fetchone()
                if shard:
                    path = directory / shard["file_path"]
                    if path.exists():
                        orphaned.append(path)
                    conn.execute("DELETE FROM shards WHERE id = ?", (shard_id,))
            orphaned.extend(
                directory / r["preview_path"] for r in rows if r["preview_path"]
            )
        # Past this line the transaction has committed: the rows are gone
        # for good, so a failed unlink is an orphan to report, never a
        # reason to resurrect them.
        for path in orphaned:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning(
                    "discard left %s behind (its rows are already deleted): %s",
                    path, exc,
                )
        return deleted

    # -- training sets -----------------------------------------------------

    def list_sets(self, name: str) -> tuple[TrainingSetInfo, ...]:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        with self._connect(directory / "metadata.db") as conn:
            rows = conn.execute(
                "SELECT ts.*, (SELECT COUNT(*) FROM set_members sm "
                "WHERE sm.set_id = ts.id) AS members "
                "FROM training_sets ts ORDER BY ts.created_at DESC"
            ).fetchall()
        return tuple(
            TrainingSetInfo(
                id=int(r["id"]),
                name=str(r["name"]),
                description=r["description"],
                created_at=_dt(r["created_at"]),
                members=int(r["members"]),
            )
            for r in rows
        )

    def commit(self, name: str, item_ids: list[int], set_name: str) -> int:
        directory = self._existing_dir(name)
        self._require_v2(directory, name)
        # Documented adapter bridge: "what commit means" (membership
        # reuse rules) lives with the trainer's manager package.
        from manager.dataset import ManagedDataset  # noqa: PLC0415

        return ManagedDataset(directory).commit_to_set(item_ids, set_name)

    # -- internals ---------------------------------------------------------

    def _validate_dir(self, name: str) -> Path:
        """Untrusted dataset name -> path that cannot escape the base."""
        if not name or not name.strip():
            raise InvalidQueryError("dataset name must not be empty")
        if name != name.strip():
            raise InvalidQueryError(
                f"dataset name must not have surrounding whitespace: {name!r}"
            )
        if "/" in name or "\\" in name or "\x00" in name:
            raise InvalidQueryError(f"invalid dataset name: {name!r}")
        if name in (".", "..") or name.startswith("."):
            raise InvalidQueryError(f"invalid dataset name: {name!r}")
        base = self._layout.datasets_dir.resolve()
        resolved = (base / name).resolve()
        if not resolved.is_relative_to(base):
            raise InvalidQueryError(f"invalid dataset name: {name!r}")
        return self._layout.datasets_dir / name

    def _existing_dir(self, name: str) -> Path:
        directory = self._validate_dir(name)
        if not (directory / "metadata.db").exists():
            raise DatasetNotFoundError(f"no dataset named '{name}'")
        return directory

    @staticmethod
    def _require_v2(directory: Path, name: str) -> None:
        version = SqliteDatasetLibrary._version(directory / "metadata.db")
        if version != 2:
            raise DatasetNotMigratedError(
                f"dataset '{name}' is in legacy format (version {version}); "
                f"run scripts/migrate_dataset_format.py on it first",
                details={"dataset": name, "format_version": version},
            )

    @staticmethod
    def _version(db: Path) -> int:
        if not db.exists():
            return 0
        conn = sqlite3.connect(str(db))
        try:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])
        finally:
            conn.close()

    def _read_info(self, directory: Path) -> DatasetInfo:
        db = directory / "metadata.db"
        with self._connect(db) as conn:
            row = conn.execute(
                "SELECT name, description, created_at FROM info"
            ).fetchone()
        created = _dt(row["created_at"]) if row else datetime.fromtimestamp(
            0, tz=timezone.utc
        )
        return DatasetInfo(
            name=str(row["name"]) if row else directory.name,
            description=(row["description"] or "") if row else "",
            created_at=created,
            format_version=self._version(db),
        )

    @staticmethod
    def _stats(directory: Path) -> DatasetStats:
        with sqlite3.connect(str(directory / "metadata.db")) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT "
                "(SELECT COUNT(*) FROM trajectories) AS items, "
                "(SELECT COUNT(*) FROM trajectories t WHERE EXISTS "
                "(SELECT 1 FROM set_members sm WHERE sm.trajectory_id = t.id)) "
                "AS committed, "
                "(SELECT COUNT(*) FROM trajectories WHERE type = 'bad') AS bad, "
                "(SELECT COUNT(*) FROM training_sets) AS sets, "
                "(SELECT COUNT(*) FROM shards) AS shards, "
                "(SELECT COALESCE(SUM(size_bytes), 0) FROM shards) AS bytes"
            ).fetchone()
        items = int(row["items"])
        committed = int(row["committed"])
        return DatasetStats(
            items=items,
            pending=items - committed,
            committed=committed,
            bad=int(row["bad"]),
            sets=int(row["sets"]),
            shards=int(row["shards"]),
            bytes=int(row["bytes"]),
        )

    @staticmethod
    def _get_item(directory: Path, item_id: int) -> DatasetItem | None:
        with sqlite3.connect(str(directory / "metadata.db")) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT t.*, (EXISTS (SELECT 1 FROM set_members sm "
                "WHERE sm.trajectory_id = t.id)) AS committed "
                "FROM trajectories t WHERE t.id = ?",
                (item_id,),
            ).fetchone()
        return _row_to_item(row) if row else None

    @contextmanager
    def _connect(self, db: Path):
        # Same pragmas as manager.db: WAL + busy_timeout so a trainer
        # child, an ingestion child, and this server can share the file.
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _row_to_item(row) -> DatasetItem:
    return DatasetItem(
        id=int(row["id"]),
        source_id=int(row["source_id"]),
        shard_id=int(row["shard_id"]),
        prompt=row["prompt"] or "",
        neg_prompt=row["neg_prompt"] or "",
        model_type=row["model_type"] or "eps",
        type=row["type"] or "good",
        cfg=row["cfg"],
        seed=int(row["seed"]) if row["seed"] is not None else None,
        source_path=row["source_path"],
        latent_h=int(row["latent_h"] or 0),
        latent_w=int(row["latent_w"] or 0),
        preview_path=row["preview_path"],
        committed=bool(row["committed"]),
    )
