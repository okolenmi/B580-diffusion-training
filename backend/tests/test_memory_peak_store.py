"""Tests for SqlitePeakStore: one writer for peaks.

The `PeakRecord` JSON file this replaces (removed in MEM-04 #3) wrote
atomically (temp file + rename) but lost updates across processes: six
processes recording distinct peaks at the same instant ended with a
stored peak lower than the highest recorded in 69 of 150 trials.

This store is the fix: a SQLite table with `INSERT ... ON CONFLICT DO
UPDATE SET peak = MAX(peak, excluded.peak)`. The ported section below
carries the 14 checks of the removed file implementation's smoke test
(monotonic, corruption reads as unknown, one fingerprint never answers
for another), so the guarantees that implementation had to keep stay
pinned on this one.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import multiprocessing
import tempfile
import threading
from pathlib import Path

from backend.infrastructure.memory_peak_store import SqlitePeakStore

#: How long the concurrency workers may take to reach (and be released
#: from) their barrier. Generous next to the real cost -- six processes
#: starting and racing takes well under a second -- because the point is
#: only to turn "hung forever" into "failed, and said so"; every trial
#: below pays it only if something is actually wrong.
_BARRIER_TIMEOUT_S = 120.0


def _store(db_path: Path) -> SqlitePeakStore:
    return SqlitePeakStore(db_path)


# -- basic record / read ----------------------------------------------------


def test_record_and_read():
    """Record a peak, read it back."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7000.0)
        assert store.reservation_mb("fp1") == 7150.0  # 7000 + 150 pillow


def test_record_monotonic():
    """A later smaller number does not lower the stored peak."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7000.0)
        store.record("fp1", 5000.0)  # smaller, should not lower
        assert store.reservation_mb("fp1") == 7150.0  # still 7000 + 150


def test_record_returns_stored():
    """record returns the stored value (the max)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        v1 = store.record("fp1", 7000.0)
        assert v1 == 7000.0
        v2 = store.record("fp1", 5000.0)
        assert v2 == 7000.0  # max, not the new value


def test_unknown_returns_none():
    """Unknown fingerprint returns None, not zero."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        assert store.reservation_mb("unknown") is None


def test_known_returns_all():
    """known() returns every remembered peak."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7000.0)
        store.record("fp2", 8000.0)
        known = store.known()
        assert known["fp1"] == 7000.0
        assert known["fp2"] == 8000.0


def test_forget():
    """forget drops one configuration's measurement."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7000.0)
        store.forget("fp1")
        assert store.reservation_mb("fp1") is None


# -- concurrent writers (the 69/150 test) ------------------------------------


def _record_worker(db_path: str, fingerprint: str, value: float, barrier):
    """Worker that records one peak after the barrier.

    The store is constructed *before* the barrier so this trial stays a
    clean test of the *write* race. Construction racing on a file that does
    not exist yet is a different hazard with its own test below
    (``test_concurrent_first_construction``): it used to delete the file
    from under the process creating it, and that was a bug in the store,
    not something for the test to be arranged around.

    The barrier is timed anyway: an untimed wait turns one wedged
    process into an indefinite hang reported only as a bare "TIMED
    OUT" by run_all's outer cap, with nothing saying which test hung.
    """
    store = SqlitePeakStore(db_path)
    barrier.wait(timeout=_BARRIER_TIMEOUT_S)
    store.record(fingerprint, value)


def test_concurrent_writers_max_semantics():
    """Six processes recording distinct peaks at the same instant:
    stored == max(values) in EVERY trial.

    This is the experiment that failed upstream (69/150 trials with the
    JSON file). With SQLite's MAX semantics, it must be 150/150.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "peaks.db")
        fingerprint = "test_fp"
        values = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
        max_value = max(values)

        trials = 30  # 30 trials x 6 processes = 180 concurrent writes
        for _ in range(trials):
            barrier = multiprocessing.Barrier(len(values))
            processes = [
                multiprocessing.Process(
                    target=_record_worker,
                    args=(db_path, fingerprint, v, barrier),
                )
                for v in values
            ]
            for p in processes:
                p.start()
            # Timed joins with stragglers reaped. An untimed join() here
            # turns one wedged worker into an indefinite hang (the shape
            # that produced the 600s "TIMED OUT" above), and leaves its
            # temp database open for the rest of the run.
            for p in processes:
                p.join(timeout=_BARRIER_TIMEOUT_S)
            stragglers = [p for p in processes if p.is_alive()]
            for p in stragglers:
                p.terminate()
                p.join(timeout=10.0)
            assert not stragglers, (
                f"{len(stragglers)} worker(s) did not finish within "
                f"{_BARRIER_TIMEOUT_S}s and had to be terminated -- the "
                f"store deadlocked rather than losing an update"
            )

            store = SqlitePeakStore(db_path)
            stored = store.reservation_mb(fingerprint, pillow_mb=0.0)
            assert stored == max_value, (
                f"stored {stored} != max {max_value} (lost update!)"
            )
            store.forget(fingerprint)  # reset for next trial


def test_concurrent_writers_threads():
    """Same test with threads (in-process concurrency)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        fingerprint = "test_fp"
        values = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
        max_value = max(values)

        barrier = threading.Barrier(len(values))
        threads = [
            threading.Thread(
                target=_record_worker,
                args=(str(Path(tmp) / "peaks.db"), fingerprint, v, barrier),
            )
            for v in values
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stored = store.reservation_mb(fingerprint, pillow_mb=0.0)
        assert stored == max_value


# -- samples counter ---------------------------------------------------------


def test_samples_incremented():
    """Each record increments the samples counter."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7000.0)
        store.record("fp1", 7000.0)
        store.record("fp1", 7000.0)
        # Read the samples count directly
        import sqlite3
        conn = sqlite3.connect(str(Path(tmp) / "peaks.db"))
        row = conn.execute(
            "SELECT samples FROM memory_peaks WHERE fingerprint = ?", ("fp1",)
        ).fetchone()
        assert row[0] == 3


# -- ported from nodes/smoke_tests/smoke_test_peak_record.py (MEM-04 #3) ----
# The numbers are ones this session measured on the B580: rank-64 LoRA at
# 1024 with checkpointing peaks at 7,666 MB at batch 2 and 8,954 MB at
# batch 4; residents are constant at 5,611 MB; reserved drift across runs
# is 0-14 MB, which is what the 150 MB pillow has to cover and why it is
# not larger. The other four of the original 14 checks live in the tests
# above: unknown is None (not zero), peak + pillow, never lowered,
# forget reads as unknown.


def test_higher_peak_raises_the_stored_value():
    """A worse run raises the mark (ported check 4)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7666.0)  # batch 2
        store.record("fp1", 8954.0)  # batch 4, measured later, is worse
        assert store.peak_mb("fp1") == 8954.0
        assert store.reservation_mb("fp1") == 9104.0


def test_fingerprints_stay_separate():
    """One configuration's numbers never answer another's (checks 5-7)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        batch2 = "sdxl|2|1024|1024|64|True|adamw"
        batch4 = "sdxl|4|1024|1024|64|True|adamw"
        no_ckpt = "sdxl|2|1024|1024|64|False|adamw"
        assert store.peak_mb(batch2) is None  # never measured: unknown
        store.record(batch2, 7666.0)
        assert store.peak_mb(batch4) is None  # another batch, unmeasured
        assert store.peak_mb(no_ckpt) is None  # another checkpointing flag
        store.record(batch4, 8954.0)
        assert store.peak_mb(batch2) == 7666.0  # recording one leaves others
        assert store.peak_mb(no_ckpt) is None


def test_store_leaves_no_litter():
    """Every write, including over a corrupt file, stays in the one file."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        store = _store(db)
        store.record("fp1", 7666.0)
        db.write_text("{ not json, not sqlite either }")
        store.record("fp1", 7666.0)  # discard + rewrite, no debris left
        store.forget("fp1")
        assert {p.name for p in Path(tmp).iterdir()} == {"peaks.db"}


def test_corrupt_db_reads_as_unknown_and_rewrites_cleanly():
    """Checks 9-10: a damaged file is a cache miss, then a clean file."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        store = _store(db)
        store.record("fp1", 7666.0)
        db.write_text("{ not json, not sqlite either }")
        # (9) corrupt reads as unknown: never a zero, never a crash.
        assert store.peak_mb("fp1") is None
        assert store.reservation_mb("fp1") is None
        assert store.known() == {}
        # (10) re-recording rewrites a clean file, and MAX starts over
        # from what is actually in it.
        assert store.record("fp1", 3000.0) == 3000.0
        assert store.peak_mb("fp1") == 3000.0
        store.record("fp1", 7666.0)
        assert store.peak_mb("fp1") == 7666.0
        # ...and a fresh instance sees the rewritten file, not the debris.
        assert _store(db).peak_mb("fp1") == 7666.0


def test_known_holds_plain_readable_numbers():
    """The rows are real numbers a person can read (check 11)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        store = _store(db)
        store.record("fp1", 7666.0)
        known = store.known()
        assert known == {"fp1": 7666.0}
        assert isinstance(known["fp1"], float)
        # ...and straight through sqlite, not only the store's own code.
        import sqlite3
        with sqlite3.connect(str(db)) as conn:
            row = conn.execute(
                "SELECT peak_mb FROM memory_peaks WHERE fingerprint = ?",
                ("fp1",),
            ).fetchone()
        assert row is not None and float(row[0]) == 7666.0


def test_forget_leaves_other_configurations_alone():
    """Forgetting one key drops only that one (check 13)."""
    with tempfile.TemporaryDirectory() as tmp:
        store = _store(Path(tmp) / "peaks.db")
        store.record("fp1", 7666.0)
        store.record("fp2", 8954.0)
        store.forget("fp1")
        assert store.peak_mb("fp1") is None
        assert store.peak_mb("fp2") == 8954.0


def test_reopen_sees_the_measurement():
    """A second store over the same file sees the first's (check 14)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        first = _store(db)
        first.record("fp1", 7666.0)
        assert _store(db).peak_mb("fp1") == 7666.0


# -- creating the store is itself a race --------------------------------------


def _construct_then_record(db_path: str, value: float, barrier, results):
    """Construct AFTER the barrier, so every process creates the file at once."""
    try:
        barrier.wait(timeout=_BARRIER_TIMEOUT_S)
        SqlitePeakStore(db_path).record("fp", value)
        results.put("ok")
    except Exception as exc:  # noqa: BLE001 -- the error text is the finding
        results.put(f"{type(exc).__name__}: {exc}")


def test_empty_file_is_a_new_database_not_damage():
    """A zero-length file is what a store looks like while it is being
    created (``sqlite3.connect`` makes the file, the first commit writes the
    header). It must survive the recovery check -- same file, same inode --
    and be usable: unknown for a read, a value after a record."""
    import os
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        db.touch()
        inode = os.stat(db).st_ino
        store = _store(db)
        assert os.stat(db).st_ino == inode, "an empty file was deleted as 'corrupt'"
        assert store.peak_mb("fp") is None
        assert store.record("fp", 7000.0) == 7000.0
        assert os.stat(db).st_ino == inode


def test_file_truncated_to_nothing_reads_as_unknown_then_recovers():
    """A cache truncated to zero bytes is a miss, not a crash."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        store = _store(db)
        store.record("fp", 7000.0)
        db.write_bytes(b"")
        assert store.peak_mb("fp") is None
        assert store.known() == {}
        assert store.record("fp", 3000.0) == 3000.0


def test_discard_skips_a_file_a_peer_already_replaced():
    """Two processes that both saw the same bad file must not both delete
    it: the second would remove the clean store the first just wrote."""
    import os
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "peaks.db"
        db.write_text("garbage, not sqlite")
        store = SqlitePeakStore.__new__(SqlitePeakStore)
        store._db_path = db
        real_stat = os.stat

        def peer_replaces_then_stat(path, *a, **kw):
            # Between this process's header read and its unlink, a peer
            # discards the bad file and writes a good one.
            if str(path) == str(db) and not getattr(peer_replaces_then_stat, "done", False):
                peer_replaces_then_stat.done = True
                db.unlink()
                SqlitePeakStore(db).record("fp", 1234.0)
            return real_stat(path, *a, **kw)

        os.stat = peer_replaces_then_stat
        try:
            assert store._discard_if_not_a_database() is False
        finally:
            os.stat = real_stat
        assert SqlitePeakStore(db).peak_mb("fp") == 1234.0, "the peer's clean store was deleted"


def test_connections_are_closed():
    """Every operation closes its connection (a bare ``with conn:`` only
    commits, leaving the handle open until garbage collection)."""
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        opened = []
        real = sqlite3.connect

        class Tracking(sqlite3.Connection):
            pass

        def tracking_connect(*a, **kw):
            kw["factory"] = Tracking
            conn = real(*a, **kw)
            opened.append(conn)
            return conn

        sqlite3.connect = tracking_connect
        try:
            store = _store(Path(tmp) / "peaks.db")
            store.record("fp", 1.0)
            store.peak_mb("fp")
            store.known()
            store.forget("fp")
        finally:
            sqlite3.connect = real
        assert opened, "the test did not observe any connection"
        for conn in opened:
            try:
                conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue
            raise AssertionError("a connection was left open")


def test_concurrent_first_construction():
    """Six processes create the store at the same instant, then record.

    Before the fix this produced ``disk I/O error``, ``no such table`` and
    lost peaks in most trials (the recovery check deleted the file another
    process was creating). Every process must finish cleanly and the
    maximum must be stored, in EVERY trial."""
    values = [1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0]
    ctx = multiprocessing.get_context()
    for trial in range(20):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = str(Path(tmp) / "peaks.db")  # does not exist yet
            barrier = ctx.Barrier(len(values))
            results = ctx.Queue()
            procs = [
                ctx.Process(target=_construct_then_record, args=(db_path, v, barrier, results))
                for v in values
            ]
            for p in procs:
                p.start()
            outcomes = [results.get(timeout=_BARRIER_TIMEOUT_S) for _ in procs]
            for p in procs:
                p.join(timeout=30.0)
            assert outcomes == ["ok"] * len(values), f"trial {trial}: {sorted(set(outcomes))}"
            assert SqlitePeakStore(db_path).peak_mb("fp") == max(values), (
                f"trial {trial}: stored peak lost"
            )


def main() -> None:
    """Run every test in this file, listed by name.

    Listed, not discovered: a `def test_*` nothing calls is a comment
    shaped like a safety net, and `scripts/check_test_wiring.py` fails
    this file when one is defined and left out here -- all 9 of
    these were, and the file exited 0 having run nothing, before that
    check caught it.
    """
    tests = [
        test_record_and_read,
        test_record_monotonic,
        test_record_returns_stored,
        test_unknown_returns_none,
        test_known_returns_all,
        test_forget,
        test_concurrent_writers_max_semantics,
        test_concurrent_writers_threads,
        test_samples_incremented,
        test_higher_peak_raises_the_stored_value,
        test_fingerprints_stay_separate,
        test_store_leaves_no_litter,
        test_corrupt_db_reads_as_unknown_and_rewrites_cleanly,
        test_known_holds_plain_readable_numbers,
        test_forget_leaves_other_configurations_alone,
        test_reopen_sees_the_measurement,
        test_empty_file_is_a_new_database_not_damage,
        test_file_truncated_to_nothing_reads_as_unknown_then_recovers,
        test_discard_skips_a_file_a_peer_already_replaced,
        test_connections_are_closed,
        test_concurrent_first_construction,
    ]
    for test in tests:
        test()
    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {len(tests)} CHECKS PASSED")


if __name__ == "__main__":
    main()
