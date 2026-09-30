#!/usr/bin/env python3
"""Migrate managed dataset(s) from format v1 to v2.

Pure SQLite + file moves: no GPU, no torch, tensor data untouched (v2 did not
change shard tensor layouts). Per dataset it:

  1. backs up metadata.db -> metadata.db.v1.bak (once),
  2. parses the v1 `trajectories.metadata` JSON blob into columns
     (neg_prompt, model_type, type, cfg, extra),
  3. derives latent_h/w from safetensors headers (torch-free header parse),
  4. flattens staging/ + archive/ files into shards/ and rewrites
     shards.file_path,
  5. gives each shard a `layout` from its rows
     (lora_raw -> single_latent, compressed -> compressed_traj, other
     format strings preserved verbatim so readers skip them),
  6. drops the legacy `tasks` table (task lifecycle belongs to the server),
  7. sets PRAGMA user_version = 2.

Usage:
    python scripts/migrate_dataset_format.py                  # whole datasets/ library
    python scripts/migrate_dataset_format.py datasets/foo     # one dataset
    python scripts/migrate_dataset_format.py datasets/        # whole library
"""

import json
import os
import shutil
import sqlite3
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

V2 = 2


def _read_header(path: Path) -> dict:
    """safetensors header alone: 8-byte little-endian u64 length, then a JSON
    dict of {name: {shape: [...]}}. No torch, no tensor data read."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            return json.loads(f.read(n))
    except FileNotFoundError:
        print(f"    warn: shard file missing: {path.name}")
    except Exception as e:  # header corrupt/unknown
        print(f"    warn: cannot read header of {path.name}: {e}")
    return {}


def _row_hw(header: dict, layout: str, shard_index: int) -> tuple[int, int]:
    """(h, w) for ONE trajectory's own tensor -- shards can mix shapes
    (variable-size crops: x0_9 may be 75x64 while x0_0 is 93x64), so this
    must be per-row, never per-file."""
    if layout == "compressed_traj":
        key = f"traj_{shard_index}_xt"
    else:
        key = f"x0_{shard_index}"
    spec = header.get(key)
    if isinstance(spec, dict) and len(spec.get("shape", [])) == 4:
        return int(spec["shape"][2]), int(spec["shape"][3])
    # Unknown layout or missing key: fall back to any latent-like tensor.
    for k, spec in header.items():
        if k == "__metadata__" or not isinstance(spec, dict):
            continue
        if len(spec.get("shape", [])) == 4:
            return int(spec["shape"][2]), int(spec["shape"][3])
    return 0, 0


def _layout_of(meta_json: str | None) -> str:
    if not meta_json:
        return "single_latent"
    try:
        m = json.loads(meta_json)
    except (json.JSONDecodeError, TypeError):
        return "single_latent"
    if not isinstance(m, dict):
        return "single_latent"
    if m.get("compressed"):
        return "compressed_traj"
    fmt = m.get("format")
    if fmt == "lora_raw" or fmt is None:
        return "single_latent"
    return str(fmt)  # legacy/unknown format string; readers skip it verbatim


def migrate_dataset(ds_dir: Path) -> bool:
    """Returns True if work was done, False if skipped/already v2."""
    db = ds_dir / "metadata.db"
    if not db.exists():
        return False

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        traj_cols = {r["name"] for r in conn.execute("PRAGMA table_info(trajectories)")}
        if "neg_prompt" in traj_cols:
            if version != V2:
                conn.execute(f"PRAGMA user_version = {V2}")
                conn.commit()
                print(f"  {ds_dir.name}: columns already v2, stamped user_version={V2}")
                return True
            print(f"  {ds_dir.name}: already v2, skipping")
            return False
        if "metadata" not in traj_cols:
            print(f"  {ds_dir.name}: UNRECOGNIZED schema (no metadata, no columns) -- "
                  f"skipping; regenerate this dataset")
            return False

        bak = ds_dir / "metadata.db.v1.bak"
        if not bak.exists():
            shutil.copy2(db, bak)
            print(f"  {ds_dir.name}: backed up -> {bak.name}")

        shards = [dict(r) for r in conn.execute("SELECT * FROM shards")]
        trajs = [dict(r) for r in conn.execute("SELECT * FROM trajectories")]
        has_tasks = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tasks'"
        ).fetchone()[0]

        # Layout per shard from its rows; headers per shard file (cached).
        shard_layout: dict[int, str] = {}
        shard_header: dict[str, dict] = {}
        traj_shard_file: dict[int, str] = {}
        for s in shards:
            rows = [t for t in trajs if t["shard_id"] == s["id"]]
            layouts = {_layout_of(t["metadata"]) for t in rows}
            shard_layout[s["id"]] = sorted(layouts)[0] if len(layouts) == 1 else (
                "single_latent" if "single_latent" in layouts else sorted(layouts)[0])
            if len(layouts) > 1:
                print(f"    warn: shard {s['file_path']} mixes layouts {layouts}; "
                      f"using {shard_layout[s['id']]}")
            traj_shard_file[s["id"]] = s["file_path"]

        # Move staging/ + archive/ files into shards/.
        moved = 0
        (ds_dir / "shards").mkdir(exist_ok=True)
        for s in shards:
            rel_old = s["file_path"]
            src = ds_dir / rel_old
            dst = ds_dir / "shards" / Path(rel_old).name
            if src.exists():
                if dst.exists() and dst != src:
                    # Distinct files with one basename can't happen (names are
                    # uuid-based), but never silently drop data if it does.
                    dst = ds_dir / "shards" / f"{src.parent.name}_{src.name}"
                os.replace(src, dst)
                moved += 1
            if not dst.exists():
                print(f"    warn: shard file missing after move: {rel_old}")
            s["_new_path"] = "shards/" + Path(rel_old).name

        # Rewrite DB: new tables alongside old, then swap.
        conn.execute("BEGIN")
        conn.execute("""
            CREATE TABLE shards_v2 (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path   TEXT    NOT NULL UNIQUE,
                layout      TEXT    NOT NULL DEFAULT 'single_latent',
                sample_count INTEGER NOT NULL DEFAULT 0,
                size_bytes  INTEGER NOT NULL DEFAULT 0,
                created_at  REAL    NOT NULL
            )""")
        conn.execute("""
            CREATE TABLE trajectories_v2 (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id   INTEGER NOT NULL,
                shard_id    INTEGER NOT NULL,
                shard_index INTEGER NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 0,
                seed        INTEGER,
                prompt      TEXT,
                neg_prompt  TEXT NOT NULL DEFAULT '',
                model_type  TEXT NOT NULL DEFAULT 'eps',
                type        TEXT NOT NULL DEFAULT 'good',
                cfg         REAL,
                source_path TEXT,
                latent_h    INTEGER NOT NULL DEFAULT 0,
                latent_w    INTEGER NOT NULL DEFAULT 0,
                preview_path TEXT,
                extra       TEXT
            )""")

        for s in shards:
            conn.execute(
                "INSERT INTO shards_v2 (id, file_path, layout, sample_count, size_bytes, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (s["id"], s["_new_path"], shard_layout.get(s["id"], "single_latent"),
                 s["sample_count"], s["size_bytes"], s["created_at"]))

        parsed_rows = 0
        for t in trajs:
            m = {}
            if t["metadata"]:
                try:
                    parsed = json.loads(t["metadata"])
                    if isinstance(parsed, dict):
                        m = parsed
                except (json.JSONDecodeError, TypeError):
                    pass
            handled = {"neg", "model_type", "type", "cfg", "format", "compressed"}
            extra = {k: v for k, v in m.items() if k not in handled}
            file_rel = traj_shard_file.get(t["shard_id"])
            h, w = 0, 0
            if file_rel:
                hw_key = "shards/" + Path(file_rel).name
                if hw_key not in shard_header:
                    shard_header[hw_key] = _read_header(ds_dir / hw_key)
                h, w = _row_hw(shard_header[hw_key],
                               shard_layout.get(t["shard_id"], "single_latent"),
                               t["shard_index"])
            conn.execute(
                "INSERT INTO trajectories_v2 (id, source_id, shard_id, shard_index, sample_count, "
                "seed, prompt, neg_prompt, model_type, type, cfg, source_path, latent_h, latent_w, "
                "preview_path, extra) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (t["id"], t["source_id"], t["shard_id"], t["shard_index"], t["sample_count"],
                 t["seed"], t["prompt"], m.get("neg") or "", m.get("model_type") or "eps",
                 m.get("type") or "good", m.get("cfg"), None, h, w,
                 t["preview_path"], json.dumps(extra) if extra else None))
            parsed_rows += 1

        conn.execute("DROP TABLE trajectories")
        conn.execute("ALTER TABLE trajectories_v2 RENAME TO trajectories")
        conn.execute("DROP TABLE shards")
        conn.execute("ALTER TABLE shards_v2 RENAME TO shards")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traj_source ON trajectories(source_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_traj_shard ON trajectories(shard_id)")
        dropped_tasks = 0
        if has_tasks:
            conn.execute("DROP TABLE tasks")
            dropped_tasks = 1
        conn.execute(f"PRAGMA user_version = {V2}")
        conn.commit()

        # Tidy now-empty legacy dirs (only if truly empty).
        for name in ("staging", "archive"):
            d = ds_dir / name
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

        print(f"  {ds_dir.name}: migrated -- {parsed_rows} trajectory row(s), "
              f"{moved} shard file(s) -> shards/, "
              f"{'tasks table dropped' if dropped_tasks else 'no tasks table'}")
        return True
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    if argv:
        targets = [Path(a).resolve() for a in argv]
    else:
        from paths import get_datasets_dir
        targets = [Path(get_datasets_dir())]

    datasets: list[Path] = []
    for t in targets:
        if (t / "metadata.db").exists():
            datasets.append(t)
        elif t.is_dir():
            datasets.extend(sorted(p for p in t.iterdir()
                                   if p.is_dir() and (p / "metadata.db").exists()))
        else:
            print(f"skip: {t} (not a dataset or library directory)")

    if not datasets:
        print("No datasets found.")
        return 1

    print(f"Migrating {len(datasets)} dataset(s) to format v{V2}...")
    did = 0
    for ds in datasets:
        try:
            if migrate_dataset(ds):
                did += 1
        except Exception as e:
            print(f"  {ds.name}: FAILED -- {e}")
            return 1
    print(f"Done: {did} dataset(s) migrated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
