"""Streaming dataset loader — reads trajectories from virtual sets."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union
import torch

from .db import get_training_set_trajectories, get_training_set_by_name
from .storage import ShardLoader
from converter.model_io import raw_to_target


class ManagedDatasetLoader:
    """Streams data from a virtual training set, with optional sample batching."""

    def __init__(self, dataset_root: Path, set_identifier: Optional[Union[int, str]] = None,
                 shuffle: bool = True, batch_size: int = 1):
        self.root = dataset_root
        self.db_path = dataset_root / "metadata.db"
        self.shuffle = shuffle
        self.batch_size = batch_size
        self._samples: list | None = None  # loaded once on first iteration, reused
        
        if set_identifier is not None:
            # Resolve set ID: accept both integer ID and string name
            if isinstance(set_identifier, str):
                resolved = get_training_set_by_name(self.db_path, set_identifier)
                if resolved is None:
                    from .db import get_training_sets
                    available = [s["name"] for s in get_training_sets(self.db_path)]
                    raise ValueError(
                        f"Training set '{set_identifier}' not found in dataset '{dataset_root.name}'. "
                        f"Available sets: {available}"
                    )
                self.set_id = resolved
            else:
                self.set_id = set_identifier
            
            # Fetch member trajectories from DB
            self.trajectories = get_training_set_trajectories(self.db_path, self.set_id)
        else:
            # Fetch ALL trajectories from DB
            from .db import get_trajectories
            self.trajectories = get_trajectories(self.db_path)
            # Normalize key names to match get_training_set_trajectories if needed
            # get_trajectories returns 'shard_path', get_training_set_trajectories returns 'file_path'
            for t in self.trajectories:
                if "file_path" not in t and "shard_path" in t:
                    t["file_path"] = t["shard_path"]
        
        # Group by shard to minimize file openings
        self.shard_map = {}
        for t in self.trajectories:
            path = t["file_path"]
            if path not in self.shard_map:
                self.shard_map[path] = []
            self.shard_map[path].append(t)

    def _load_all_samples(self) -> list:
        """Load every sample from every trajectory into a flat list.

        Samples from different trajectories and timesteps are interleaved
        so that subsequent shuffling produces a truly random order across
        both images and timesteps.  The list is held in RAM; for typical
        datasets (100-1000 images × 20 timesteps) this is well under 1 GB.
        """
        all_samples = []
        for path, trajs in self.shard_map.items():
            loader = ShardLoader(self.root / path)
            loader.load()
            try:
                for t in trajs:
                    neg_prompt = ""
                    meta = {}
                    trajectory_samples = []

                    if t.get("metadata"):
                        try:
                            meta = json.loads(t["metadata"])
                            neg_prompt = meta.get("neg", "")

                            if meta.get("compressed"):
                                raw_samples = loader.get_compressed_trajectory(t["shard_index"])
                                cfg_val = meta.get("cfg", 7.5)
                                m_type = meta.get("model_type", "eps")
                                if "model_type" not in meta:
                                    import warnings
                                    warnings.warn(
                                        "Dataset trajectory has no 'model_type' in metadata — "
                                        "assuming 'eps'.  If the teacher was a vpred model, the "
                                        "stored targets are wrong and this data should be regenerated "
                                        "with the updated run_teacher_task (model_type='vpred').",
                                        stacklevel=2,
                                    )
                                for s in raw_samples:
                                    if s["t"] == 0:
                                        continue
                                    p = s["target_p"]
                                    n = s["target_n"]
                                    s["target"] = n + (p - n) * cfg_val
                                    trajectory_samples.append(s)
                            else:
                                trajectory_samples = loader.get_trajectory_samples(
                                    t["shard_index"], t["sample_count"])
                        except (json.JSONDecodeError, TypeError):
                            trajectory_samples = loader.get_trajectory_samples(
                                t["shard_index"], t["sample_count"])
                    else:
                        trajectory_samples = loader.get_trajectory_samples(
                            t["shard_index"], t["sample_count"])

                    traj_type = meta.get("type", "good") if isinstance(meta, dict) else "good"
                    for s in trajectory_samples:
                        if s["t"] == 0:
                            continue
                        all_samples.append({
                            "x_t":      s["x_t"],
                            "target":   s["target"],
                            "target_p": s.get("target_p"),
                            "target_n": s.get("target_n"),
                            "t":        s["t"],
                            "prompt":      t["prompt"],
                            "neg_prompt":  neg_prompt,
                            "seed":        t["seed"],
                            "metadata":    t["metadata"],
                            "traj_type":   traj_type,
                        })
            finally:
                loader.close()
        return all_samples

    def _iter_samples(self) -> Iterator[Dict]:
        """Yield individual unbatched samples, shuffled across all trajectories and timesteps.

        The dataset is loaded from disk exactly once and kept in RAM.  Subsequent
        epochs shuffle the in-memory list in-place — no disk access after the
        first load.  This eliminates the per-epoch stall that previously caused
        the prefetcher queue to drain and the GPU to idle every epoch boundary.

        Tensors are NOT pinned here — pinning is deferred to __iter__ so it
        happens one batch at a time rather than all samples upfront.
        """
        if self._samples is None:
            print("  [DataLoader] Loading dataset into RAM...")
            self._samples = self._load_all_samples()
            print(f"  [DataLoader] {len(self._samples)} samples loaded.")

        if self.shuffle:
            random.shuffle(self._samples)

        yield from self._samples

    @staticmethod
    def _merge_samples(samples: list) -> Dict:
        """Merge individual samples into a single batched dict."""
        out = {
            "x_t": torch.cat([s["x_t"] for s in samples], dim=0),
            "target": torch.cat([s["target"] for s in samples], dim=0),
            "t": torch.tensor([s["t"] for s in samples]),
            "prompt": samples[0]["prompt"],
            "neg_prompt": samples[0]["neg_prompt"],
            "seed": samples[0]["seed"],
            "metadata": samples[0]["metadata"],
            "traj_type": samples[0]["traj_type"],
        }
        if samples[0].get("target_p") is not None:
            out["target_p"] = torch.cat([s["target_p"] for s in samples], dim=0)
        if samples[0].get("target_n") is not None:
            out["target_n"] = torch.cat([s["target_n"] for s in samples], dim=0)
        return out

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over samples, yielding batches of batch_size with shared prompt and size.

        Implements a bucketing strategy:
        1. Groups all samples into buckets by (prompt, neg_prompt, size).
        2. Shuffles samples within each bucket.
        3. Forms full batches from buckets.
        4. Groups batches of the same shape into 'clumps' (e.g. 4 batches of same shape).
        5. Shuffles the clumps to maintain global randomness while minimizing
           expensive GPU kernel switches between different shapes.
        """
        if self._samples is None:
            print("  [DataLoader] Loading dataset into RAM...")
            self._samples = self._load_all_samples()
            print(f"  [DataLoader] {len(self._samples)} samples loaded.")

        # 1. Group by key (prompt, neg_prompt, size)
        buckets = {}
        for s in self._samples:
            size = s["x_t"].shape[2:]
            key = (s["prompt"], s["neg_prompt"], size)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(s)

        # 2. Shuffle within buckets and form batches
        all_batches = []
        for key, samples in buckets.items():
            if self.shuffle:
                random.shuffle(samples)
            
            for i in range(0, len(samples), self.batch_size):
                chunk = samples[i : i + self.batch_size]
                # Drop incomplete last batch if shuffling (common training practice)
                if len(chunk) < self.batch_size and self.shuffle:
                    continue
                all_batches.append(self._merge_samples(chunk))

        if not all_batches:
            return

        if not self.shuffle:
            for b in all_batches:
                yield self._pin_batch(b)
            return

        # 3. Clump batches of the same shape together to minimize kernel switches.
        # This is the "66 kernels" fix: instead of switching shape every step,
        # we process a small clump of the same shape, then switch.
        CLUMP_SIZE = 4
        clumps = []
        # Re-group batches by their size key
        shape_buckets = {}
        for b in all_batches:
            size = b["x_t"].shape[2:]
            if size not in shape_buckets:
                shape_buckets[size] = []
            shape_buckets[size].append(b)
        
        for size_batches in shape_buckets.values():
            random.shuffle(size_batches)
            for i in range(0, len(size_batches), CLUMP_SIZE):
                clumps.append(size_batches[i : i + CLUMP_SIZE])
        
        # 4. Shuffle the clumps and yield
        random.shuffle(clumps)
        for clump in clumps:
            for batch in clump:
                yield self._pin_batch(batch)

    @staticmethod
    def _pin_batch(batch: Dict) -> Dict:
        """Pin all tensor values in a batch dict for fast non-blocking GPU transfer."""
        return {
            k: v.pin_memory() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def invalidate_cache(self):
        """Force the next iteration to reload from disk (e.g. after dataset update)."""
        self._samples = None

    def __len__(self):
        return sum(t["sample_count"] for t in self.trajectories)
