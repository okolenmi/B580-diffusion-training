"""SqliteDatasetLibrary tests -- real adapter over temp dataset dirs.

Run directly: python backend/tests/test_dataset_library.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.application.errors import (
    DatasetAlreadyExistsError,
    DatasetFileNotFoundError,
    DatasetItemNotFoundError,
    DatasetNotFoundError,
    DatasetNotMigratedError,
    InvalidQueryError,
)
from backend.application.ports.dataset_library import BulkItemChanges, ItemChanges
from backend.infrastructure.dataset_files import FsDatasetFiles
from backend.infrastructure.dataset_library import SqliteDatasetLibrary
from backend.infrastructure.workspace import WorkspaceLayout
from backend.tests.support import (
    check,
    finish,
    make_v1_dataset,
    make_v2_dataset,
)

root = Path(tempfile.mkdtemp(prefix="backend-dataset-lib-"))
library = SqliteDatasetLibrary(WorkspaceLayout(root))
datasets = root / "datasets"


def expect(exc_type, fn, label):
    try:
        fn()
    except exc_type:
        check(True, label)
    else:
        check(False, f"{label} (expected {exc_type.__name__})")


# -- listing ---------------------------------------------------------------

check(library.list() == (), "empty library lists nothing")

# -- create (manager bridge: torch + byte-identical schema) ----------------

info = library.create("bridge-ds", "created through the bridge")
check(info.name == "bridge-ds", "create returns the name")
check(info.description == "created through the bridge", "create stores description")
check(info.format_version == 2, "bridge-created dataset is format v2")
check((datasets / "bridge-ds" / "shards").is_dir(), "create makes shards/")
check((datasets / "bridge-ds" / "previews").is_dir(), "create makes previews/")
check(library.get("bridge-ds").description == "created through the bridge", "get round-trips")

expect(DatasetAlreadyExistsError, lambda: library.create("bridge-ds"), "duplicate create refused")

(datasets / "ghost").mkdir(parents=True)
(datasets / "ghost" / "junk.txt").write_text("leftover")
ghost_info = library.create("ghost")
check(ghost_info.format_version == 2, "ghost dir (no metadata.db) is wiped and recreated")
check(not (datasets / "ghost" / "junk.txt").exists(), "ghost content removed")

# -- raw fixture: summaries / stats ----------------------------------------

make_v2_dataset(root, "raw", items=4)
by_name = {s.info.name: s for s in library.list()}
check(set(by_name) == {"bridge-ds", "ghost", "raw"}, "list sees all three datasets")
check(by_name["raw"].stats is not None, "v2 dataset has stats")
check(by_name["bridge-ds"].stats is not None and by_name["bridge-ds"].stats.items == 0,
      "fresh dataset stats are zeroed, not None")

stats = library.stats("raw")
check(stats.items == 4, "stats.items counts rows")
check(stats.pending == 4 and stats.committed == 0, "stats starts fully pending")
check(stats.bad == 1, "stats.bad counts type='bad'")
check(stats.shards == 1 and stats.bytes == 1024, "stats shards/bytes from shards table")
check(stats.sets == 0, "stats.sets starts empty")

# -- items: read + filter ---------------------------------------------------

items = library.list_items("raw")
check(len(items) == 4, "list_items returns every row")
check(items[0].prompt == "photo 1", "prompt column read")
check(items[0].neg_prompt == "", "neg_prompt column read (default '')")
check(items[-1].type == "bad", "last fixture row is bad")
check(items[0].latent_h == 64 and items[0].latent_w == 64, "latent dims read")
check(all(not it.committed for it in items), "nothing committed yet")

# -- single update ----------------------------------------------------------

updated = library.update_item("raw", 2, ItemChanges(prompt="new prompt"))
check(updated.prompt == "new prompt", "single update writes prompt")
check(library.update_item("raw", 2, ItemChanges(type="bad")).type == "bad",
      "single update flips type")
check(library.update_item("raw", 2, ItemChanges(prompt="")).prompt == "",
      "empty string clears prompt (None would not)")
check(library.update_item("raw", 2, ItemChanges(neg_prompt="soft")).neg_prompt == "soft",
      "single update writes neg_prompt")
check(library.update_item("raw", 3, ItemChanges(cfg=1.5)).cfg == 1.5, "single update writes cfg")
expect(DatasetItemNotFoundError,
       lambda: library.update_item("raw", 999, ItemChanges(cfg=1.0)),
       "unknown item id refused")

# -- bulk update ------------------------------------------------------------

check(library.bulk_update("raw", [1, 2, 3], BulkItemChanges(cfg=7.5)) == 3,
      "bulk update matches all ids")
by_id = {it.id: it for it in library.list_items("raw")}
check(all(by_id[i].cfg == 7.5 for i in (1, 2, 3)), "bulk cfg applied")
check(by_id[4].cfg == 5.0, "unlisted row untouched")

library.bulk_update("raw", [1], BulkItemChanges(prompt="warm "))
check({it.id: it for it in library.list_items("raw")}[1].prompt == "warm ",
      "bulk set mode replaces prompt")
library.bulk_update("raw", [1], BulkItemChanges(prompt="[trigger]", prompt_mode="prepend"))
check({it.id: it for it in library.list_items("raw")}[1].prompt.startswith("[trigger]"),
      "bulk prepend mode prefixes prompt")
check(library.bulk_update("raw", [1], BulkItemChanges(prompt="")) == 1,
      "bulk with empty prompt still matches")
check({it.id: it for it in library.list_items("raw")}[1].prompt.startswith("[trigger]"),
      "bulk empty prompt is a no-op (legacy truthy gate)")

# -- sets + commit (manager bridge) ----------------------------------------

set_id = library.commit("raw", [1, 2], "first set")
check(set_id >= 1, "commit returns a set id")
sets = library.list_sets("raw")
check(len(sets) == 1 and sets[0].name == "first set", "set listed by name")
check(sets[0].members == 2, "two members committed")
check(library.commit("raw", [3], "first set") == set_id, "commit reuses set by name")
check(library.list_sets("raw")[0].members == 3, "membership grows across commits")

check(len(library.list_items("raw", committed=True)) == 3, "committed filter")
check(len(library.list_items("raw", committed=False)) == 1, "pending filter")
stats = library.stats("raw")
check(stats.committed == 3 and stats.pending == 1, "stats reflect membership")

# -- discard ---------------------------------------------------------------

raw_dir = datasets / "raw"
check((raw_dir / "shards" / "x0_0.safetensors").exists(), "shard file present")
check(library.discard("raw", [4]) == 1, "discard deletes one row")
check(len(library.list_items("raw")) == 3, "row gone")
check((raw_dir / "shards" / "x0_0.safetensors").exists(),
      "shard file kept while rows remain")

check(library.discard("raw", [1]) == 1, "discard committed row")
check(library.list_sets("raw")[0].members == 2, "membership cascades with the row")
check(not (raw_dir / "previews" / "p1.png").exists(), "preview file unlinked")

check(library.discard("raw", [2, 3]) == 2, "discard last rows")
check(not (raw_dir / "shards" / "x0_0.safetensors").exists(),
      "empty shard file unlinked")
check(library.stats("raw").shards == 0, "empty shard row removed")

# -- legacy (v1) refusal ----------------------------------------------------

make_v1_dataset(root, "legacy")
by_name = {s.info.name: s for s in library.list()}
check("legacy" in by_name, "v1 dataset still listed")
check(by_name["legacy"].info.format_version == 0, "v1 reported with its real version")
check(by_name["legacy"].stats is None, "v1 stats are None (never fabricated)")
check(library.get("legacy").format_version == 0, "v1 identity readable")
expect(DatasetNotMigratedError, lambda: library.stats("legacy"),
       "v1 stats refused with migration error")
expect(DatasetNotMigratedError, lambda: library.list_items("legacy"),
       "v1 items refused")
expect(DatasetNotMigratedError, lambda: library.list_sets("legacy"),
       "v1 sets refused")
expect(DatasetNotMigratedError,
       lambda: library.commit("legacy", [1], "s"), "v1 commit refused")
check(library.delete("legacy") is True, "v1 delete allowed without migrating")
check("legacy" not in {s.info.name for s in library.list()}, "v1 gone from list")

# -- name validation + not-found --------------------------------------------

for bad in ("", "   ", "../evil", "a/b", "a\\b", ".", "..", ".hidden", " padded "):
    expect(InvalidQueryError, lambda b=bad: library.get(bad),
           f"invalid name {bad!r} refused")
expect(DatasetNotFoundError, lambda: library.get("nope"), "get unknown -> not found")
expect(DatasetNotFoundError, lambda: library.root("nope"), "root unknown -> not found")
check(library.delete("nope") is False, "delete unknown -> False")

# -- file serving (M8c): scoped byte reads ----------------------------------

# fresh dataset: earlier sections deliberately unlinked raw's preview
make_v2_dataset(root, "file-ds", items=2)
files = FsDatasetFiles(datasets)
blob = files.read("file-ds", "previews/p1.png")
check(blob.content.startswith(b"\x89PNG"), "preview bytes round-trip exactly")
check(blob.media_type == "image/png", "png media type from suffix")
expect(DatasetNotFoundError, lambda: files.read("nope", "previews/p1.png"),
       "files.read unknown dataset -> not found")
expect(DatasetNotFoundError, lambda: files.read("../..", "x.png"),
       "escaping dataset name -> not found")
expect(DatasetFileNotFoundError, lambda: files.read("file-ds", "previews/absent.png"),
       "missing file -> not found")
expect(DatasetFileNotFoundError, lambda: files.read("file-ds", "../../escape.txt"),
       "traversal out of the dataset dir refused")

# -- item fetch + preview fallback (M8f) ------------------------------------

make_v2_dataset(root, "pv", items=4)  # item 1 previews p1.png, item 4 is bad
item = library.get_item("pv", 1)
check(item.id == 1 and item.preview_path == "previews/p1.png" and item.type == "good",
      "get_item round-trips the row")
expect(DatasetItemNotFoundError, lambda: library.get_item("pv", 99),
       "get_item unknown id -> not found")
expect(DatasetNotFoundError, lambda: library.get_item("nope", 1),
       "get_item unknown dataset -> not found")

check(library.first_preview("pv") == "previews/p1.png",
      "first_preview = first non-bad row carrying an image")
check(library.first_preview("nope") is None,
      "first_preview unknown dataset -> None (best effort)")
make_v1_dataset(root, "pv-legacy")
check(library.first_preview("pv-legacy") is None,
      "first_preview legacy dataset -> None (v2 columns untouched)")
expect(DatasetNotMigratedError, lambda: library.get_item("pv-legacy", 1),
       "get_item on a legacy dataset -> not migrated")

# a bad-only image must never front the dataset
conn = sqlite3.connect(str(datasets / "pv" / "metadata.db"))
try:
    conn.execute("UPDATE trajectories SET preview_path = NULL WHERE id = 1")
    conn.execute("UPDATE trajectories SET preview_path = 'previews/p4.png' WHERE id = 4")
    conn.commit()
finally:
    conn.close()
check(library.first_preview("pv") is None,
      "first_preview skips the bad row's image (honest null)")
check(library.get_item("pv", 4).preview_path == "previews/p4.png",
      "get_item still reads the bad row's own preview")

finish()
