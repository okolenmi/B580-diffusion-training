"""Unified Dataset Core — managed state, storage, and curation lifecycle."""

import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

from .db import (
    _connect, init_local_db, set_dataset_info,
    get_trajectories, delete_trajectory, delete_shard, get_training_sets,
    get_active_tasks, update_task_status
)
from .storage import ShardLoader, ShardWriter
from core.model_io import raw_to_target


class ManagedDataset:
    """Represents a single self-contained dataset directory."""

    def __init__(self, root: Path):
        self.root = root
        self.db_path = root / "metadata.db"
        self.staging_dir = root / "staging"
        self.archive_dir = root / "archive"
        self.preview_dir = root / "previews"
        
        # Ensure directory structure
        for d in [self.staging_dir, self.archive_dir, self.preview_dir]:
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

    # --- Staging (Inbox) ---

    def get_staging_trajectories(self, source_id: int = None) -> List[dict]:
        """Get all trajectories currently in the staging area."""
        return get_trajectories(self.db_path, is_temp=True, source_id=source_id)

    def get_archived_trajectories(self, source_id: int = None) -> List[dict]:
        """Get all trajectories that have been committed to the archive."""
        return get_trajectories(self.db_path, is_temp=False, source_id=source_id)

    # --- Curation ---

    def toggle_trajectory_type(self, traj_id: int):
        """Toggle trajectory type between 'good' and 'bad'."""
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT metadata FROM trajectories WHERE id = ?", (traj_id,)).fetchone()
            if not row:
                raise ValueError(f"Trajectory {traj_id} not found")
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            current = meta.get("type", "good")
            meta["type"] = "bad" if current == "good" else "good"
            conn.execute("UPDATE trajectories SET metadata = ? WHERE id = ?",
                         (json.dumps(meta), traj_id))
            conn.commit()

    def update_trajectory(self, traj_id: int, prompt: str = None, neg_prompt: str = None, cfg: float = None):
        """Update trajectory prompt, negative prompt, and/or CFG in metadata."""
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT prompt, metadata FROM trajectories WHERE id = ?", (traj_id,)).fetchone()
            if not row:
                raise ValueError(f"Trajectory {traj_id} not found")
            
            new_prompt = prompt if prompt is not None else row["prompt"]
            meta = json.loads(row["metadata"]) if row["metadata"] else {}
            
            if neg_prompt is not None:
                meta["neg"] = neg_prompt
            if cfg is not None:
                meta["cfg"] = cfg
                
            conn.execute("UPDATE trajectories SET prompt = ?, metadata = ? WHERE id = ?",
                         (new_prompt, json.dumps(meta), traj_id))
            conn.commit()

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
        """Physically move trajectories from Staging to Archive and create a training set."""
        if not traj_ids:
            return 0

        # 1. Fetch trajectory + shard info
        with _connect(self.db_path) as conn:
            rows = conn.execute(f"""
                SELECT t.id, t.shard_id, t.shard_index, t.sample_count, t.seed,
                       t.prompt, t.preview_path, t.metadata,
                       s.file_path as shard_path, s.is_temporary
                FROM trajectories t
                JOIN shards s ON t.shard_id = s.id
                WHERE t.id IN ({','.join(['?']*len(traj_ids))})
            """, traj_ids).fetchall()
            trajs = [dict(r) for r in rows]

        if len(trajs) != len(traj_ids):
            print(f"  Warning: {len(traj_ids) - len(trajs)} trajectory(s) not found, continuing with {len(trajs)}.")

        # Group by source shard
        shard_traj_map: dict[int, list[dict]] = {}
        for t in trajs:
            shard_traj_map.setdefault(t["shard_id"], []).append(t)
        source_shard_ids = set(shard_traj_map.keys())

        # Check if ALL trajectories in each source shard are being committed
        can_move = True
        with _connect(self.db_path) as conn:
            for sid, group in shard_traj_map.items():
                total = conn.execute(
                    "SELECT COUNT(*) FROM trajectories WHERE shard_id = ?", (sid,)
                ).fetchone()[0]
                if total != len(group):
                    can_move = False
                    break

        if can_move:
            # --- MOVE PATH: rename file, no tensor rewrite ---
            new_rel_paths: dict[int, str] = {}
            for sid in source_shard_ids:
                t = shard_traj_map[sid][0]
                old_rel = t["shard_path"]
                new_rel = f"archive/set_{uuid.uuid4().hex[:16]}.safetensors"
                old_abs = self.root / old_rel
                new_abs = self.root / new_rel
                new_abs.parent.mkdir(parents=True, exist_ok=True)
                os.rename(str(old_abs), str(new_abs))
                new_rel_paths[sid] = new_rel

            with _connect(self.db_path) as conn:
                for sid, new_rel in new_rel_paths.items():
                    conn.execute(
                        "UPDATE shards SET file_path = ?, is_temporary = 0 WHERE id = ?",
                        (new_rel, sid)
                    )

                now = time.time()
                cur = conn.execute(
                    "INSERT INTO training_sets (name, description, created_at) VALUES (?, ?, ?)",
                    (set_name, None, now)
                )
                set_id = cur.lastrowid

                for t in trajs:
                    conn.execute(
                        "INSERT INTO set_members (set_id, trajectory_id) VALUES (?, ?)",
                        (set_id, t["id"])
                    )
                conn.commit()
            return set_id

        # --- COPY PATH: extract, rewrite, cleanup ---
        shard_rel_path = f"archive/set_{uuid.uuid4().hex[:16]}.safetensors"
        shard_abs_path = self.root / shard_rel_path
        writer = ShardWriter(shard_abs_path)

        updated_data = []
        loaders = {}

        try:
            for t in trajs:
                path = self.root / t["shard_path"]
                if path not in loaders:
                    loaders[path] = ShardLoader(path)

                samples = []
                m_raw = t["metadata"]
                is_compressed = False
                if m_raw:
                    try:
                        m_data = json.loads(m_raw)
                        is_compressed = m_data.get("compressed", False)
                    except (json.JSONDecodeError, TypeError):
                        pass

                if is_compressed:
                    raw_samples = loaders[path].get_compressed_trajectory(t["shard_index"])
                    cfg_val = m_data.get("cfg", 7.5)
                    samples = []
                    for s in raw_samples:
                        p = s["target_p"]
                        n = s["target_n"]
                        raw = n + (p - n) * cfg_val
                        at = torch.tensor([s["at"]]).view(1, 1, 1, 1)
                        st = torch.tensor([s["st"]]).view(1, 1, 1, 1)
                        s["target"] = raw_to_target(raw, s["x_t"], at, st, "eps", "eps")
                        samples.append(s)
                else:
                    samples = loaders[path].get_trajectory_samples(t["shard_index"], t["sample_count"])

                new_idx = writer.add_trajectory(samples)
                updated_data.append((new_idx, t["id"]))
        finally:
            for l in loaders.values(): l.close()

        sample_count, size_bytes = writer.write()

        with _connect(self.db_path) as conn:
            cur = conn.execute(
                "INSERT INTO shards (file_path, sample_count, size_bytes, is_temporary, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (shard_rel_path, sample_count, size_bytes, 0, time.time())
            )
            new_shard_id = cur.lastrowid

            now = time.time()
            cur = conn.execute(
                "INSERT INTO training_sets (name, description, created_at) VALUES (?, ?, ?)",
                (set_name, None, now)
            )
            set_id = cur.lastrowid

            for new_idx, traj_id in updated_data:
                conn.execute(
                    "UPDATE trajectories SET shard_id = ?, shard_index = ? WHERE id = ?",
                    (new_shard_id, new_idx, traj_id)
                )
                conn.execute(
                    "INSERT INTO set_members (set_id, trajectory_id) VALUES (?, ?)",
                    (set_id, traj_id)
                )

            for sid in source_shard_ids:
                count = conn.execute(
                    "SELECT COUNT(*) FROM trajectories WHERE shard_id = ?", (sid,)
                ).fetchone()[0]
                if count == 0:
                    row = conn.execute(
                        "SELECT file_path FROM shards WHERE id = ?", (sid,)
                    ).fetchone()
                    if row:
                        file_path = self.root / row["file_path"]
                        if file_path.exists():
                            file_path.unlink()
                        conn.execute("DELETE FROM shards WHERE id = ?", (sid,))

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
