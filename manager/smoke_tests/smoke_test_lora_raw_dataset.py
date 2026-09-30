"""Real sqlite + safetensors shard round trip (temp dir, no mocks) for the
lora_raw format added to manager/storage.py + manager/loader.py: verifies
a trajectory stored as a single clean latent (a) reads back correctly,
and (b) gets a genuinely fresh (x_t, target, t) every __iter__() call --
not the same one reused, which was the entire point.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from core.noise_schedule import eps_to_x0
from manager.db import add_shard, add_source, init_local_db
from manager.loader import ManagedDatasetLoader
from manager.storage import ShardLoader, ShardWriter


def check_shard_round_trip(tmpdir: Path):
    print("[ShardWriter/ShardLoader: add_image_latent round trip]")
    x0 = torch.randn(1, 4, 8, 8)
    shard_file = tmpdir / "test.safetensors"
    writer = ShardWriter(shard_file)
    idx = writer.add_image_latent(x0)
    writer.write()

    loader = ShardLoader(shard_file)
    loaded = loader.get_image_latent(idx)
    torch.testing.assert_close(loaded, x0)
    print("    PASS")


def _no_gpu_pin_memory_workaround():
    """This sandbox has no GPU driver at all, and ManagedDatasetLoader._pin_batch
    unconditionally calls .pin_memory() (pre-existing behavior, unrelated to
    what's being tested here). Patches it out for the duration of this test
    only -- not a production code change."""
    torch.Tensor.pin_memory = lambda self, *a, **kw: self


def check_fresh_resampling_each_iteration(tmpdir: Path):
    print("[ManagedDatasetLoader: lora_raw trajectory resamples fresh every __iter__()]")
    dataset_root = tmpdir / "dataset"
    dataset_root.mkdir()
    db_path = dataset_root / "metadata.db"
    init_local_db(db_path)

    x0 = torch.randn(1, 4, 8, 8)
    shard_file = dataset_root / "shards" / "shard.safetensors"
    writer = ShardWriter(shard_file)
    idx = writer.add_image_latent(x0)
    count, size = writer.write()

    source_id = add_source(db_path, "test_source", "real")
    shard_id = add_shard(db_path, str(shard_file.relative_to(dataset_root)), count, size)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, prompt, "
        "neg_prompt, model_type, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, shard_id, idx, 1, 1, "a cat", "", "eps", "good"),
    )
    conn.commit()
    conn.close()

    loader = ManagedDatasetLoader(dataset_root, shuffle=False, batch_size=1)
    batch1 = next(iter(loader))
    batch2 = next(iter(loader))  # second __iter__() call -- must resample, not reuse

    assert "x_t" in batch1 and "target" in batch1 and "t" in batch1
    assert not torch.equal(batch1["x_t"], batch2["x_t"]) or batch1["t"][0].item() != batch2["t"][0].item(), \
        "two separate __iter__() calls produced identical noise+timestep -- resampling isn't fresh"

    for batch in (batch1, batch2):
        at, st = None, None
        from core.noise_schedule import get_alpha_sigma
        at, st = get_alpha_sigma(batch["t"])
        x0_recovered = eps_to_x0(batch["target"], batch["x_t"],
                                  at.view(-1, 1, 1, 1), st.view(-1, 1, 1, 1))
        torch.testing.assert_close(x0_recovered, x0, atol=1e-5, rtol=1e-4)
    print("    PASS: two epochs got different (x_t, t), and both recover the exact same x0")


def _make_one_sample_dataset(tmpdir: Path, name: str = "dataset_single") -> Path:
    """A dataset with exactly one real sample -- shared setup for the
    regression check below, the exact-mode and skip checks, and
    (implicitly) the resampling check above's same shape, factored out
    since several tests need the identical one-sample setup (each caller
    passes its own `name` so the roots never collide)."""
    dataset_root = tmpdir / name
    dataset_root.mkdir()
    db_path = dataset_root / "metadata.db"
    init_local_db(db_path)

    x0 = torch.randn(1, 4, 8, 8)
    shard_file = dataset_root / "shards" / "shard.safetensors"
    writer = ShardWriter(shard_file)
    idx = writer.add_image_latent(x0)
    count, size = writer.write()

    source_id = add_source(db_path, "test_source", "real")
    shard_id = add_shard(db_path, str(shard_file.relative_to(dataset_root)), count, size)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, prompt, "
        "neg_prompt, model_type, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (source_id, shard_id, idx, 1, 1, "a cat", "", "eps", "good"),
    )
    conn.commit()
    conn.close()
    return dataset_root


def check_undersized_dataset_raises_not_silently_empty(tmpdir: Path):
    """Regression check for a real, reported bug, not a hypothesis: a
    1-sample dataset with batch_size=2 and shuffle=True (the default)
    used to make __iter__() silently yield zero batches, every epoch,
    forever -- the incomplete last chunk of every bucket gets dropped
    when shuffling (correct, intentional, in general), but when a
    bucket's samples are *entirely* one incomplete chunk, dropping it
    drops everything. That surfaced several frames away in real training
    (nodes/train/step_pipeline.py's FetchBatchPhase: one caught
    StopIteration to wrap to a new epoch, then an uncaught second one
    immediately after, crashing the run with a bare "StopIteration" and
    no indication why) -- reproduced directly against the real class
    here instead, checking the fix at its actual source."""
    print("[ManagedDatasetLoader: undersized dataset raises clearly, doesn't silently "
          "yield zero batches forever]")
    dataset_root = _make_one_sample_dataset(tmpdir)

    loader = ManagedDatasetLoader(dataset_root, shuffle=True, batch_size=2)
    try:
        list(loader)
        raise AssertionError("expected ValueError, got no error -- fix regressed")
    except ValueError as e:
        assert "no batch can ever be formed" in str(e), f"wrong error message: {e}"
        print(f"    PASS: raised a clear ValueError instead of silently yielding "
              f"zero batches: {str(e)[:80]}...")

    # A real dataset genuinely having zero samples must still just yield
    # nothing, unchanged -- this fix only fires when there WAS data that
    # got entirely dropped, not for a truly empty dataset.
    loader2 = ManagedDatasetLoader(dataset_root, shuffle=True, batch_size=2)
    loader2._samples = []
    assert list(loader2) == [], "a genuinely empty dataset should still just yield nothing"
    print("    PASS: a genuinely empty dataset (0 samples) still yields nothing, "
          "not an error -- unchanged")

    # The same 1-sample dataset with a batch_size it CAN satisfy must be
    # completely unaffected by this fix.
    loader3 = ManagedDatasetLoader(dataset_root, shuffle=True, batch_size=1)
    batches = list(loader3)
    assert len(batches) == 1, f"expected 1 batch, got {len(batches)}"
    print("    PASS: the same dataset with batch_size=1 (satisfiable) is unaffected")


def check_exact_mode_round_trip(tmpdir: Path):
    print("[ManagedDatasetLoader: t_mode='exact' pins t to t_values and cycles it]")
    dataset_root = _make_one_sample_dataset(tmpdir, name="dataset_exact")
    loader = ManagedDatasetLoader(dataset_root, shuffle=False, batch_size=1,
                                  t_mode="exact", t_values="100,200")
    t1 = next(iter(loader))["t"][0].item()
    t2 = next(iter(loader))["t"][0].item()   # cursor persists across __iter__ (epochs)
    t3 = next(iter(loader))["t"][0].item()
    assert (t1, t2, t3) == (100, 200, 100), \
        f"expected cycle [100, 200, 100] across epochs, got [{t1}, {t2}, {t3}]"
    print(f"    PASS: t pinned and cycled across epochs ({t1}, {t2}, {t3})")


def check_non_single_latent_trajectories_skipped(tmpdir: Path):
    print("[ManagedDatasetLoader: non-single-latent shards are skipped, not misread]")
    dataset_root = _make_one_sample_dataset(tmpdir, name="dataset_mixed")
    db_path = dataset_root / "metadata.db"
    # A teacher/compressed trajectory in its own shard (v2: layout is a
    # per-shard property, one file = one layout). The loader must skip the
    # whole shard (honestly reported) instead of trying to read it as a
    # clean latent -- and the single real sample still trains.
    import sqlite3
    writer = ShardWriter(dataset_root / "shards" / "teacher.safetensors")
    traj_id = writer.add_compressed_trajectory(
        torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8), torch.randn(3, 4, 8, 8),
        [999, 500, 1], [{"at": 1.0, "st": 0.0}] * 3)
    count, size = writer.write()
    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT source_id FROM trajectories").fetchone()
    shard2 = add_shard(db_path, "shards/teacher.safetensors", count, size,
                       layout="compressed_traj")
    conn.execute(
        "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, prompt, "
        "neg_prompt, model_type, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (row[0], shard2, traj_id, 3, 1, "a cat", "", "eps", "good"),
    )
    conn.commit()
    conn.close()

    loader = ManagedDatasetLoader(dataset_root, shuffle=False, batch_size=1)
    batches = list(loader)
    assert len(batches) == 1, \
        f"expected only the single-latent sample to survive, got {len(batches)} batches"
    print("    PASS: the compressed/teacher shard was skipped; the "
          "single-latent sample still yields its batch")


def main():
    _no_gpu_pin_memory_workaround()
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        check_shard_round_trip(tmpdir)
        check_fresh_resampling_each_iteration(tmpdir)
        check_undersized_dataset_raises_not_silently_empty(tmpdir)
        check_exact_mode_round_trip(tmpdir)
        check_non_single_latent_trajectories_skipped(tmpdir)
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
