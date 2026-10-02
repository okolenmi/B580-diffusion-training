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
    DatasetDirectoryConflictError,
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

check(library.list_datasets() == (), "empty library lists nothing")

# -- create (manager bridge: torch + byte-identical schema) ----------------

info = library.create("bridge-ds", "created through the bridge")
check(info.name == "bridge-ds", "create returns the name")
check(info.description == "created through the bridge", "create stores description")
check(info.format_version == 2, "bridge-created dataset is format v2")
check((datasets / "bridge-ds" / "shards").is_dir(), "create makes shards/")
check((datasets / "bridge-ds" / "previews").is_dir(), "create makes previews/")
check(library.get("bridge-ds").description == "created through the bridge", "get round-trips")

expect(DatasetAlreadyExistsError, lambda: library.create("bridge-ds"), "duplicate create refused")

# A ghost directory (no metadata.db) is only cleared when it is the empty
# skeleton create() makes. A directory with content may be the user's own
# image folder that happens to share the name -- this server never deletes
# files it did not create (docs 07 F-10).
(datasets / "ghost").mkdir(parents=True)
ghost_info = library.create("ghost")
check(ghost_info.format_version == 2, "empty ghost dir is recreated")
check((datasets / "ghost" / "shards").is_dir(), "recreated ghost has the skeleton")

(datasets / "mine").mkdir(parents=True)
(datasets / "mine" / "IMG_0001.png").write_bytes(b"\x89PNG")
expect(
    DatasetDirectoryConflictError,
    lambda: library.create("mine"),
    "a directory with content is refused",
)
check(
    (datasets / "mine" / "IMG_0001.png").exists(),
    "the user's own file was not deleted",
)
check(
    not (datasets / "mine" / "metadata.db").exists(),
    "and no dataset was written into it either",
)

# -- raw fixture: summaries / stats ----------------------------------------

make_v2_dataset(root, "raw", items=4)
by_name = {s.info.name: s for s in library.list_datasets()}
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

# -- discard ordering: rows commit first (docs 07 F-05) --------------------
#
# Unlinking a shard inside the transaction and then failing left the
# dataset pointing at a file that was already gone. Now the rows commit
# first, so a failed unlink can only leave an orphan file -- recoverable
# -- never a dangling row.

import logging  # noqa: E402 -- local to this scenario
import shutil  # noqa: E402

make_v2_dataset(root, "ordered", items=2)
ordered_dir = datasets / "ordered"
ordered_shard = ordered_dir / "shards" / "x0_0.safetensors"

records: list[str] = []


class _Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())


capture = _Capture()
library_logger = logging.getLogger("backend.infrastructure.dataset_library")
library_logger.addHandler(capture)

real_unlink = Path.unlink


def _failing_unlink(self, *args, **kwargs):
    if self.name == ordered_shard.name and self.parent == ordered_shard.parent:
        raise OSError("injected: read-only filesystem")
    return real_unlink(self, *args, **kwargs)


Path.unlink = _failing_unlink
try:
    check(library.discard("ordered", [1, 2]) == 2, "discard of the last rows succeeded")
finally:
    Path.unlink = real_unlink
    library_logger.removeHandler(capture)

check(library.list_items("ordered") == (), "rows are gone (the commit stands)")
check(
    library.stats("ordered").shards == 0,
    "the shard row went with its last row -- no row points at a missing file",
)
check(ordered_shard.exists(), "the undeletable shard survived as an orphan file")
check(
    any(ordered_shard.name in message for message in records),
    f"the leftover is reported, not swallowed (got {records})",
)

# -- the orphan sweep (docs 08 N-12) ---------------------------------------
#
# An orphan file is the acceptable outcome of the ordering above, but
# nothing ever retried it: the row is gone, so no future discard would
# look at that shard again. These checks pin that the sweep removes it,
# that a file a row still references is never touched, and that one
# locked file does not stop the sweep of the rest.

make_v2_dataset(root, "sweep", items=2)
sweep_dir = datasets / "sweep"
kept_shard = sweep_dir / "shards" / "referenced.safetensors"
loose_shard = sweep_dir / "shards" / "loose.safetensors"

# Give one item a row that still points at a real shard file.
with library._connect(sweep_dir / "metadata.db") as conn:  # noqa: SLF001
    conn.execute(
        "INSERT INTO shards (id, file_path, created_at) VALUES "
        "(99, 'shards/referenced.safetensors', ?)",
        ("2026-10-02 00:00:00",),
    )
kept_shard.write_bytes(b"still-referenced")
loose_shard.write_bytes(b"nobody-references-this")

removed = library.sweep_orphan_shards("sweep")
check(removed == 1, f"exactly the unreferenced shard is removed (got {removed})")
check(not loose_shard.exists(), "the orphan is gone")
check(kept_shard.exists(), "a file a row still references is never removed")

# Idempotent: a second sweep finds nothing to do.
check(library.sweep_orphan_shards("sweep") == 0, "a second sweep is a no-op")

# One locked file must not stop the others being cleaned up.
blocked = sweep_dir / "shards" / "blocked.safetensors"
freeable = sweep_dir / "shards" / "freeable.safetensors"
blocked.write_bytes(b"locked")
freeable.write_bytes(b"freeable")
real_unlink2 = Path.unlink


def _failing_unlink2(self, *args, **kwargs):
    if self.name == blocked.name:
        raise OSError("injected: locked")
    return real_unlink2(self, *args, **kwargs)


Path.unlink = _failing_unlink2
try:
    removed = library.sweep_orphan_shards("sweep")
finally:
    Path.unlink = real_unlink2

check(removed == 1, f"the sweep continues past a locked file (got {removed})")
check(not freeable.exists(), "and still removes the ones it can")
check(blocked.exists(), "the locked one is left for next time")
check(kept_shard.exists(), "and the referenced shard is still untouched")

# A dataset with no shards/ directory is not an error.
make_v2_dataset(root, "nosweeps", items=1)
shutil.rmtree(datasets / "nosweeps" / "shards")
check(library.sweep_orphan_shards("nosweeps") == 0,
      "a dataset with no shards/ sweeps cleanly")

# -- legacy (v1) refusal ----------------------------------------------------

make_v1_dataset(root, "legacy")
by_name = {s.info.name: s for s in library.list_datasets()}
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
check("legacy" not in {s.info.name for s in library.list_datasets()},
      "v1 gone from list")

# -- name validation + not-found --------------------------------------------

for bad in ("", "   ", "../evil", "a/b", "a\\b", ".", "..", ".hidden", " padded "):
    # The body uses `b`, the bound parameter -- not `bad`. Binding it and
    # then reading the loop variable is the bug ruff B023 exists to catch:
    # every iteration would have asserted against the *last* value, and
    # this passed only because every value in the tuple is invalid.
    expect(InvalidQueryError, lambda b=bad: library.get(b),
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

# -- paging: a dataset is the one collection whose size is unbounded ------
#
# It grows by ingestion, not by user action, so "return every row" grew
# the response until the browser stopped rendering it -- and the UI had
# no way to know it had only seen part of the data (docs 07 F-14,
# docs 08 Q10). These checks pin the default page, the total, and that
# paging covers every row exactly once.

make_v2_dataset(root, "big", items=1200, previews=False)

from backend.application.use_cases.list_dataset_items import (  # noqa: E402
    ListDatasetItems,
)

items_use_case = ListDatasetItems(library=library)

# count_items at the adapter: the whole filtered result, not a page.
check(library.count_items("big") == 1200,
      f"count_items counts every row (got {library.count_items('big')})")
check(library.count_items("big", committed=False) == 1200,
      "every row is pending")
check(library.count_items("big", committed=True) == 0,
      "and none is committed")

# The default page, without asking for one.
default_page = items_use_case.execute("big")
check(len(default_page.items) == 500,
      f"the default page is 500 rows (got {len(default_page.items)})")
check(default_page.limit == 500, f"and says so (got {default_page.limit})")
check(default_page.total == 1200,
      f"total describes the whole result (got {default_page.total})")
check(default_page.next_offset == 500,
      f"next_offset points at the next page (got {default_page.next_offset})")
check(default_page.items[0].id == 1 and default_page.items[-1].id == 500,
      "the page is the first 500 by id")

# An explicit limit is honoured, and an explicit offset moves the window.
second = items_use_case.execute("big", limit=500, offset=500)
check(second.items[0].id == 501,
      f"offset 500 starts at row 501 (got {second.items[0].id})")
check(second.next_offset == 1000, f"next_offset advances (got {second.next_offset})")

# The last page says so, rather than offering an offset that returns
# nothing -- an empty page mid-dataset must not be mistaken for the end.
last = items_use_case.execute("big", limit=500, offset=1000)
check(len(last.items) == 200, f"the last page has the remainder (got {len(last.items)})")
check(last.next_offset is None, f"the last page reports no next (got {last.next_offset})")

# Paging covers all 1,200 exactly once: no gap, no duplicate.
seen: list[int] = []
offset = 0
pages = 0
while True:
    page = items_use_case.execute("big", offset=offset)
    seen.extend(item.id for item in page.items)
    pages += 1
    if page.next_offset is None:
        break
    offset = page.next_offset
check(pages == 3, f"1,200 rows take 3 default pages (got {pages})")
check(len(seen) == 1200, f"paging returned every row (got {len(seen)})")
check(len(set(seen)) == 1200, f"and none twice ({len(seen) - len(set(seen))} duplicates)")
check(seen == sorted(seen), "and in id order")
check(seen == list(range(1, 1201)), "with no gap in the ids")

# A filtered count is the count of that filter, not of everything -- or a
# client cannot tell "no more rows" from "no rows match".
pending = items_use_case.execute("big", committed=False, limit=10)
check(pending.total == 1200, f"pending filter total (got {pending.total})")
check(pending.next_offset == 10, f"and it still knows there is more (got {pending.next_offset})")
none_committed = items_use_case.execute("big", committed=True, limit=10)
check(none_committed.total == 0, f"no committed rows -> total 0 (got {none_committed.total})")
check(none_committed.next_offset is None,
      f"and no next page even though a limit was given (got {none_committed.next_offset})")

finish()
