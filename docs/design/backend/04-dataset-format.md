# Dataset Format v2

Status: implemented (pre-M3b; the format was changed *before* adapting the new
server, while regeneration/migration was still cheap).

## Why v2 exists

The v1 format worked for training but fought every server built on top of it:

1. **Stringly-typed metadata.** `trajectories.metadata` was an unvalidated JSON
   blob (`neg`, `format`, `model_type`, `type`, `cfg`, `compressed`,
   `batch_idx`, …) — the entire query surface for listing/editing/filtering
   required `json_extract`/Python parsing, and every writer had to remember the
   key names.
2. **Staging/archive duality (`shards.is_temporary`).** Real datasets lived in
   `staging/` forever; the commit path's COPY branch read `x_t_*` keys from
   single-latent shards that store `x0_*` (crash on any *partial* commit), and
   the split made "committed" a physical property (file location) instead of a
   logical one (set membership).
3. **`tasks` table inside the dataset DB.** Background-task state (including
   PIDs) is *server* state, not dataset state: a copied dataset carried foreign
   task rows, startup reconciliation would have to reach into every dataset,
   and the table had no WAL/locking story.
4. **No concurrency PRAGMAs** — documented `database is locked` failures
   between the server writing and a trainer child reading.
5. **No size or provenance columns** — latent shape was only implied by tensor
   data (server listings had to open shard files), and the origin image path
   was discarded at ingestion time.

## On-disk layout (v2)

```
datasets/<name>/
├── metadata.db      # SQLite, PRAGMA user_version = 2
├── shards/          # all safetensors shards (was staging/ + archive/)
└── previews/        # *.webp thumbnails, trajectory rows reference them relatively
```

## SQLite schema (user_version = 2)

```sql
info(name TEXT PRIMARY KEY, description TEXT, created_at REAL)

sources(id, name, type, model_path, config, created_at)          -- unchanged

shards(
    id INTEGER PRIMARY KEY,
    file_path   TEXT NOT NULL UNIQUE,   -- relative to dataset root, "shards/..."
    layout      TEXT NOT NULL DEFAULT 'single_latent',
        -- 'single_latent'      : x0_{i} clean latents (LoRA images+captions)
        -- 'compressed_traj'    : traj_{i}_{xt,p,n,t} denoising sequences
        -- anything else        : legacy/unknown; readers must skip, not guess
    sample_count, size_bytes, created_at)

trajectories(
    id INTEGER PRIMARY KEY,
    source_id, shard_id, shard_index,    -- unchanged
    sample_count, seed, prompt,          -- unchanged
    neg_prompt  TEXT NOT NULL DEFAULT '',
    model_type  TEXT NOT NULL DEFAULT 'eps',   -- 'eps' | 'vpred'
    type        TEXT NOT NULL DEFAULT 'good',  -- 'good' | 'bad' (curation flag)
    cfg         REAL,                          -- teacher-only per-row CFG
    source_path TEXT,                          -- origin image, relative to the
                                               -- ingestion image_dir (NULL for
                                               -- migrated v1 rows and teacher rows)
    latent_h    INTEGER NOT NULL DEFAULT 0,    -- from the shard tensor shape
    latent_w    INTEGER NOT NULL DEFAULT 0,
    preview_path TEXT,
    extra       TEXT)                          -- residual JSON (e.g. crop_idx,
                                               -- batch_idx); never read by core

training_sets(id, name UNIQUE, description, created_at)          -- unchanged
set_members(set_id, trajectory_id)                               -- unchanged

-- tasks table: REMOVED. Task lifecycle belongs to the server (backend.db).
```

Connection PRAGMAs on every open: `journal_mode=WAL`, `busy_timeout=5000`,
`foreign_keys=ON`.

## Semantics changes

- **Training sets are membership only.** `commit_to_set(traj_ids, name)`
  inserts `set_members` rows (reusing the set when the name already exists) —
  no file movement, no re-sharding. The v1 MOVE/COPY paths are deleted, which
  also deletes the broken COPY branch by construction. A shard file never moves
  after ingestion; discarding trajectories may delete a shard only once it has
  no rows left.
- **Pending vs. committed is a view, not a location**: pending = not a member
  of any training set; committed = member of ≥ 1 set.
- **Layout is a property of the shard, not the row** (one shard file has one
  key layout); readers gate on `shards.layout`.
- **Format version = `PRAGMA user_version`** (2). Readers that require v2 call
  `manager.db.ensure_v2` and fail with migration guidance on v1 datasets.

## Unchanged (deliberately)

- **Tensor layouts** (`x0_{i}`, `traj_{i}_{*}`) — no re-encoding needed;
  migration is DB-only plus file moves.
- **Loader batch semantics**: bucket by (prompt, neg_prompt, size), shared
  caption per batch, incomplete-group dropping + `keep_incomplete`, t-sampling
  (`manager/t_sampling.py`), batch dict keys (`x_t`, `target`, `t`, `prompt`,
  `neg_prompt`, `seed`, `metadata`, `traj_type`).
- **Ingestion behavior**: captions from `.txt` sidecars, crop splitting with
  `max_aspect_ratio`, preview generation, teacher `(prompt, seed)` dedup.

## Transition rules

| actor \ dataset | v1 (legacy) | v2 |
|---|---|---|
| legacy server `server/` curation UI | works unchanged | works (staging/archived views are membership-based aliases; rows carry a synthesized `metadata` JSON string projected from columns for the UI) |
| legacy server ingestion/tasks | works, but `ensure_v2` in the builder refuses → task marked `failed` with migration hint | refuses at `create_task` (`no such table: tasks`) — replaced by the backend's dataset tasks (M3b, implemented) |
| training (`core.cli` → `manager.loader`) | refuses via `ensure_v2` with migration hint | works |
| new backend `backend/` | refuses via `ensure_v2` at the port boundary | works |

Migration: `scripts/migrate_dataset_format.py [dir ...]` (default: the
`datasets/` library). Pure SQLite + file moves, no GPU, backs up
`metadata.db.v1.bak` first, parses the v1 JSON blob into columns, derives
`latent_h/w` from shard headers (no torch), flattens `staging/`+`archive/`
into `shards/`, drops `tasks`, sets `user_version=2`. Old datasets can instead
be regenerated from raw images — that remains valid.

Removed on purpose from v1 → v2: dedup on re-ingest (recorded provenance
`source_path` makes it possible later), mixed-caption batching (trainer-side,
separate concern), loader RAM strategy (future work, enabled by size columns).

## M3b contract implications (implemented)

- `DatasetLibrary` port reads with its own SQL over the v2 columns (no
  `json_extract`), torch-free; `create`/`commit` bridge to `manager`
  lazily (byte-identical schema, one definition of membership).
- Dataset task lifecycle lives in `backend.db` (migration `004`); ingestion
  children report through a duck-typed reporter (`progress/finished/failed`)
  instead of writing a dataset DB `tasks` table — the "fork task gateway".
- The dataset **card preview pointer** follows the same rule (M8f): the
  stored override lives in `backend.db` (migration `006`, removed with
  the dataset), while the fallback (first non-bad item's `preview_path`)
  and the override's existence check are read-only over `trajectories` —
  a dataset directory never stores UI state.
- Startup reconciliation of dataset tasks mirrors `ReconcileRuns`, and
  the next task start sweeps too (one `DatasetTaskSweeper` serves both);
  the task list is a pure query and writes nothing.
