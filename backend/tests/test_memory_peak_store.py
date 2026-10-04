"""Tests for SqlitePeakStore: one writer for peaks.

The existing PeakRecord (nodes/memory/peak_record.py) writes atomically
(temp file + rename) but loses updates across processes: six processes
recording distinct peaks at the same instant ended with a stored peak
lower than the highest recorded in 69 of 150 trials.

This module is the fix: a SQLite table with `INSERT ... ON CONFLICT DO
UPDATE SET peak = MAX(peak, excluded.peak)`.
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
    """Worker that records one peak after the barrier."""
    store = SqlitePeakStore(db_path)
    barrier.wait()
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
            for p in processes:
                p.join()

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
    ]
    for test in tests:
        test()
    print()
    print("=" * 60)
    print(f"SMOKE TEST: ALL {len(tests)} CHECKS PASSED")


if __name__ == "__main__":
    main()
