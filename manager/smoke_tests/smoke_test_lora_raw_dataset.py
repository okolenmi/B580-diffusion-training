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
    shard_file = dataset_root / "staging" / "shard.safetensors"
    writer = ShardWriter(shard_file)
    idx = writer.add_image_latent(x0)
    count, size = writer.write()

    source_id = add_source(db_path, "test_source", "real")
    shard_id = add_shard(db_path, str(shard_file.relative_to(dataset_root)), count, size)

    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, prompt, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (source_id, shard_id, idx, 1, 1, "a cat",
         json.dumps({"neg": "", "format": "lora_raw", "model_type": "eps"})),
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


def main():
    _no_gpu_pin_memory_workaround()
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        check_shard_round_trip(tmpdir)
        check_fresh_resampling_each_iteration(tmpdir)
    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
