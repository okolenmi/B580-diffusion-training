"""Unified Dataset Core — managed state, storage, and curation lifecycle."""

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from .db import (
    _connect, init_local_db, set_dataset_info,
    get_trajectories, delete_trajectory, delete_shard, get_training_sets,
    get_active_tasks
)


class ManagedDataset:
    """Represents a single self-contained dataset directory."""

    def __init__(self, root: Path):
        self.root = root
        self.db_path = root / "metadata.db"
        self.shards_dir = root / "shards"
        self.preview_dir = root / "previews"

        # Ensure directory structure
        for d in [self.shards_dir, self.preview_dir]:
            d.mkdir(parents=True, exist_ok=True)

        if not self.db_path.exists():
            init_local_db(self.db_path)

    @property
    def name(self) -> str:
        return self.root.name

    def get_info(self) -> dict:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM info").fetchone()
            if row:
                info = dict(row)
                return {
                    "name": info.get("name", self.name),
                    "description": info.get("description", ""),
                    "created_at": info.get("created_at", 0),
                }
            return {"name": self.name, "description": "", "created_at": 0}

    # --- Views (v2: pending vs. committed is membership, not location) ---

    def get_staging_trajectories(self, source_id: int = None) -> List[dict]:
        """Legacy view name: trajectories not yet in any training set."""
        return get_trajectories(self.db_path, committed=False, source_id=source_id)

    def get_archived_trajectories(self, source_id: int = None) -> List[dict]:
        """Legacy view name: trajectories committed to at least one training set."""
        return get_trajectories(self.db_path, committed=True, source_id=source_id)

    def list_trajectories(self, source_id: int = None, committed: bool = None) -> List[dict]:
        """All trajectories (committed=None), or filtered by set membership."""
        return get_trajectories(self.db_path, committed=committed, source_id=source_id)

    # --- Curation ---

    def toggle_trajectory_type(self, traj_id: int):
        """Toggle trajectory type between 'good' and 'bad'."""
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT type FROM trajectories WHERE id = ?", (traj_id,)).fetchone()
            if not row:
                raise ValueError(f"Trajectory {traj_id} not found")
            new_type = "bad" if (row["type"] or "good") == "good" else "good"
            conn.execute("UPDATE trajectories SET type = ? WHERE id = ?", (new_type, traj_id))
            conn.commit()

    def update_trajectory(self, traj_id: int, prompt: str = None, neg_prompt: str = None, cfg: float = None):
        """Update trajectory prompt, negative prompt, and/or CFG columns."""
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT prompt, neg_prompt, cfg FROM trajectories WHERE id = ?", (traj_id,)).fetchone()
            if not row:
                raise ValueError(f"Trajectory {traj_id} not found")

            conn.execute(
                "UPDATE trajectories SET prompt = ?, neg_prompt = ?, cfg = ? WHERE id = ?",
                (prompt if prompt is not None else row["prompt"],
                 neg_prompt if neg_prompt is not None else row["neg_prompt"],
                 cfg if cfg is not None else row["cfg"],
                 traj_id)
            )
            conn.commit()

    def update_trajectories_bulk(self, traj_ids: List[int], prompt: str = None, prompt_mode: str = "set",
                                  neg_prompt: str = None, cfg: float = None) -> int:
        """Bulk-update prompt/neg_prompt/cfg across many trajectories in one transaction --
        e.g. apply a universal trigger word or a common CFG value across an entire dataset
        (or a selected subset) in a single action, instead of editing trajectories one by one.

        prompt_mode:
          'set'     -- overwrite each selected trajectory's prompt entirely with `prompt`.
          'prepend' -- add `prompt` in front of each trajectory's existing prompt (skips a
                       trajectory if its prompt already starts with it, so re-applying the
                       same trigger word twice is a safe no-op instead of duplicating it).
                       Useful when some trajectories already have per-image captions you
                       don't want to lose, and you just want to add a shared trigger word.

        `neg_prompt`/`cfg` are always a plain overwrite (there's no equivalent "prepend"
        concept for those). Any of prompt/neg_prompt/cfg left as None (or prompt="") is left
        untouched on every selected trajectory.

        Returns the number of trajectories actually updated.
        """
        if not traj_ids:
            return 0
        updated = 0
        with _connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(traj_ids))
            rows = conn.execute(
                f"SELECT id, prompt, neg_prompt, cfg FROM trajectories WHERE id IN ({placeholders})",
                traj_ids
            ).fetchall()
            for row in rows:
                new_prompt = row["prompt"] or ""
                if prompt:
                    if prompt_mode == "prepend":
                        if not new_prompt.startswith(prompt):
                            new_prompt = f"{prompt} {new_prompt}".strip()
                    else:  # "set"
                        new_prompt = prompt

                conn.execute(
                    "UPDATE trajectories SET prompt = ?, neg_prompt = ?, cfg = ? WHERE id = ?",
                    (new_prompt,
                     neg_prompt if neg_prompt else row["neg_prompt"],
                     cfg if cfg is not None else row["cfg"],
                     row["id"])
                )
                updated += 1
            conn.commit()
        return updated

    def discard_trajectories(self, traj_ids: List[int]):
        """Permanently delete trajectories and their preview files."""
        for tid in traj_ids:
            # Find preview path first
            with _connect(self.db_path) as conn:
                row = conn.execute("SELECT preview_path, shard_id FROM trajectories WHERE id = ?", (tid,)).fetchone()
                if not row: continue
                
                # Delete preview
                if row["preview_path"]:
                    p = self.root / row["preview_path"]
                    if p.exists(): p.unlink()
                
                shard_id = row["shard_id"]
                
            # Delete from DB
            delete_trajectory(self.db_path, tid)
            
            # Try to cleanup shard if it's now empty
            delete_shard(self.db_path, shard_id)

    def commit_to_set(self, traj_ids: List[int], set_name: str) -> int:
        """Commit trajectories to a named training set (membership only).

        v2: no files move. A shard is written once at ingestion and stays put;
        "committed" means "member of at least one training set". Re-committing
        to an existing set name adds members to that set instead of creating a
        duplicate name (the loader resolves sets by name, so uniqueness is a
        correctness requirement, not cosmetics).
        """
        if not traj_ids:
            return 0

        with _connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id FROM trajectories WHERE id IN (%s)" % ",".join("?" * len(traj_ids)),
                traj_ids
            ).fetchall()
            if len(row) != len(traj_ids):
                print(f"  Warning: {len(traj_ids) - len(row)} trajectory(s) not found, "
                      f"committing {len(row)}.")

            existing = conn.execute(
                "SELECT id FROM training_sets WHERE name = ?", (set_name,)
            ).fetchone()
            if existing:
                set_id = existing["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO training_sets (name, description, created_at) VALUES (?, ?, ?)",
                    (set_name, None, time.time())
                )
                set_id = cur.lastrowid

            for r in row:
                conn.execute(
                    "INSERT OR IGNORE INTO set_members (set_id, trajectory_id) VALUES (?, ?)",
                    (set_id, r["id"])
                )
            conn.commit()

        return set_id


    # --- Training Sets ---

    def get_sets(self) -> List[dict]:
        return get_training_sets(self.db_path)

    # --- Tasks ---

    def get_active_tasks(self) -> List[dict]:
        return get_active_tasks(self.db_path)


class ManagedDatasetLibrary:
    """Manages the collection of all datasets in a root directory."""

    def __init__(self, library_root: Path):
        self.root = library_root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> List[dict]:
        """List all valid datasets in the library."""
        results = []
        if not self.root.exists(): return []
        
        for d in sorted(self.root.iterdir()):
            if d.is_dir() and (d / "metadata.db").exists():
                ds = ManagedDataset(d)
                info = ds.get_info()
                results.append({
                    "name": d.name,
                    "description": info.get("description", ""),
                    "created_at": info.get("created_at", 0)
                })
        return results

    def get_dataset(self, name: str) -> ManagedDataset:
        path = self.root / name
        if not path.exists():
            raise ValueError(f"Dataset '{name}' does not exist.")
        return ManagedDataset(path)

    def create_dataset(self, name: str, description: str = None) -> ManagedDataset:
        path = self.root / name
        if path.exists():
            # Check if it's a ghost directory (no metadata.db)
            if not (path / "metadata.db").exists():
                print(f"  Cleaning up ghost directory: {path}")
                shutil.rmtree(path)
            else:
                raise ValueError(f"Dataset '{name}' already exists.")
        
        path.mkdir(parents=True)
        ds = ManagedDataset(path)
        set_dataset_info(path / "metadata.db", name, description)
        return ds

    def delete_dataset(self, name: str):
        path = self.root / name
        if not path.exists(): return
        
        # Security check
        if not path.resolve().is_relative_to(self.root.resolve()):
             raise ValueError("Security violation: attempt to delete outside dataset root.")

        shutil.rmtree(path)
