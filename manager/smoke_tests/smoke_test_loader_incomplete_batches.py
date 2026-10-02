"""ManagedDatasetLoader's handling of incomplete (prompt, size) groups.

Batches are formed per identical (prompt, neg_prompt, size) group. With
shuffle=True the incomplete last chunk of each group used to be dropped, so
with per-image captions and batch_size > 1 every image in a group smaller
than batch_size was never trained on -- silently (the only guard fired when
*nothing* survived). Real sqlite + shard round trip, no mocks:

  * default (keep_incomplete=False): the historical behavior is unchanged --
    singleton-caption images are still never yielded -- but a one-time
    warning with the exact counts is printed;
  * keep_incomplete=True: every image is yielded every epoch (remainders as
    smaller batches), and no warning is printed;
  * shuffle=False and batch_size=1 never warn (nothing is dropped).
"""

import contextlib
import io
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from nodes.components.noise_schedule import eps_to_x0, get_alpha_sigma
from manager.db import add_shard, add_source, init_local_db
from manager.loader import ManagedDatasetLoader
from manager.storage import ShardWriter

torch.Tensor.pin_memory = lambda self, *a, **kw: self  # no GPU driver in the sandbox


def check(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


PROMPTS = ["a", "a", "a", "b", "c", "d", "e", "e"]   # groups: a:3  b:1  c:1  d:1  e:2


def _make_dataset(tmpdir: Path, latents: list) -> Path:
    root = tmpdir / "ds"
    root.mkdir()
    db = root / "metadata.db"
    init_local_db(db)
    shard_file = root / "shards" / "shard.safetensors"
    writer = ShardWriter(shard_file)
    idxs = [writer.add_image_latent(x0) for x0 in latents]
    count, size = writer.write()
    source_id = add_source(db, "src", "real")
    shard_id = add_shard(db, str(shard_file.relative_to(root)), count, size)
    conn = sqlite3.connect(str(db))
    for i, (idx, prompt) in enumerate(zip(idxs, PROMPTS)):
        conn.execute(
            "INSERT INTO trajectories (source_id, shard_id, shard_index, sample_count, seed, "
            "prompt, neg_prompt, model_type, type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (source_id, shard_id, idx, 1, i, prompt, "", "eps", "good"))
    conn.commit()
    conn.close()
    return root


def _images_seen(loader, latents, epochs: int):
    """Which source images appear in each epoch, recovered from x0 = x_t - sigma*eps."""
    per_epoch = []
    for _ in range(epochs):
        seen = set()
        with contextlib.redirect_stdout(io.StringIO()):
            batches = list(loader)
        for b in batches:
            a, s = get_alpha_sigma(b["t"])
            x0 = eps_to_x0(b["target"], b["x_t"], a.view(-1, 1, 1, 1), s.view(-1, 1, 1, 1))
            for row in x0:
                matches = [i for i, l in enumerate(latents)
                           if torch.allclose(row, l[0], atol=1e-4, rtol=1e-4)]
                check(len(matches) == 1, f"could not attribute a sample to one source image: {matches}")
                seen.add(matches[0])
        per_epoch.append(seen)
    return per_epoch


def _capture(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue()


def main():
    torch.manual_seed(0)
    latents = [torch.randn(1, 4, 8, 8) for _ in PROMPTS]
    with tempfile.TemporaryDirectory() as td:
        root = _make_dataset(Path(td), latents)

        print("[default: singleton-caption images still dropped, but announced once]")
        loader = ManagedDatasetLoader(root, shuffle=True, batch_size=2)
        out = _capture(lambda: list(loader))
        check("WARNING" in out and "3 of 8 samples" in out and "NEVER trained on" in out, out)
        check("only 4 of 8" in out, out)          # a:2 of 3 + e:2 of 2 = 4 used
        check("1 more are skipped" in out, out)   # a: 3 % 2
        check(_capture(lambda: list(loader)).count("WARNING") == 0, "warning must print once")
        seen = _images_seen(loader, latents, epochs=40)
        never = set(range(8)) - set().union(*seen)
        # images 3,4,5 are the b/c/d singletons
        check(never == {3, 4, 5}, f"expected exactly the singleton-caption images never seen, got {never}")
        check(all(len(s) == 4 for s in seen), [len(s) for s in seen])
        print("    PASS")

        print("[keep_incomplete=True: every image, every epoch; no warning]")
        loader = ManagedDatasetLoader(root, shuffle=True, batch_size=2, keep_incomplete=True)
        check("WARNING" not in _capture(lambda: list(loader)), "no warning when nothing is dropped")
        seen = _images_seen(loader, latents, epochs=10)
        check(all(s == set(range(8)) for s in seen), seen)
        sizes = sorted(len(b["t"]) for b in loader)
        check(sizes == [1, 1, 1, 1, 2, 2], sizes)   # b,c,d singles; a's leftover; a-pair; e-pair
        print("    PASS")

        print("[shuffle=False and batch_size=1 never drop, never warn]")
        for kwargs in (dict(shuffle=False, batch_size=2), dict(shuffle=True, batch_size=1)):
            loader = ManagedDatasetLoader(root, **kwargs)
            check("WARNING" not in _capture(lambda: list(loader)), kwargs)
            seen = _images_seen(loader, latents, epochs=3)
            check(all(s == set(range(8)) for s in seen), (kwargs, seen))
        print("    PASS")
    print("ALL PASS")


if __name__ == "__main__":
    main()
