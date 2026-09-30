"""Streaming dataset loader — reads trajectories from virtual sets."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union
import torch

from .db import get_training_set_trajectories, get_training_set_by_name
from .storage import ShardLoader
from .t_sampling import TrainTimeSampler
from core.noise_schedule import eps_to_vpred, get_alpha_sigma


class ManagedDatasetLoader:
    """Streams data from a virtual training set, with optional sample batching."""

    def __init__(self, dataset_root: Path, set_identifier: Optional[Union[int, str]] = None,
                 shuffle: bool = True, batch_size: int = 1, use_dataset_cfg: bool = True,
                 t_low: int = 1, t_high: int = 999, t_mode: str = "uniform",
                 bucket_balance=None, t_values: str = "",
                 keep_incomplete: bool = False):
        self.root = dataset_root
        self.db_path = dataset_root / "metadata.db"
        self.shuffle = shuffle
        self.batch_size = batch_size
        # Kept as a parameter for core/trainer.py's existing call site; the
        # only code that ever read it (baked-format dual-pass target
        # blending) went out with that format -- single-latent batches carry
        # no stored target_p/target_n for it to gate.
        self.use_dataset_cfg = use_dataset_cfg
        # Batches are formed per (prompt, neg_prompt, size) group, and with
        # shuffle=True an incomplete last chunk of each group is dropped
        # (see __iter__). With per-image captions that silently removes
        # whole groups: a group smaller than batch_size never produces a
        # batch, so its images are never trained on. keep_incomplete=True
        # emits those chunks as smaller batches instead. Default False =
        # the historical behavior, unchanged (and announced once, below).
        self.keep_incomplete = keep_incomplete
        self._warned_dropping = False
        # Single-latent datasets: every t is chosen at draw time, so this is
        # the whole of a run's t configuration. Interpreted and validated in
        # one place (manager/t_sampling.py) before any DB access, so a
        # misconfigured graph fails as a config error, not mid-iteration:
        # "adaptive without a balance", "exact without/with invalid values",
        # unknown modes, empty ranges. The bucket_balance is duck-typed on
        # purpose -- manager/ must not import nodes/ -- documented by
        # contract in t_sampling.py, not by type here.
        self.t_low = t_low
        self.t_high = t_high
        self.t_mode = t_mode
        self._t_sampler = TrainTimeSampler(t_mode, t_low, t_high,
                                           bucket_balance=bucket_balance,
                                           t_values=t_values)
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
        """Load every single-latent ("lora_raw") sample into a flat list.

        One entry per image -- ingestion stores one clean latent, so there
        is no per-timestep structure left to interleave (the docstring this
        replaced described the retired baked-grid format). Trajectories in
        any other format (teacher/compressed sequences, or shards written
        by the retired run_ingestion_task) are skipped, not guessed at:
        they are not single-latent data, and reading them as such would be
        wrong. The list is held in RAM; for typical datasets (100-1000
        images) this is well under 1 GB.
        """
        all_samples = []
        skipped = {}
        for path, trajs in self.shard_map.items():
            loader = ShardLoader(self.root / path)
            loader.load()
            try:
                for t in trajs:
                    meta = {}
                    if t.get("metadata"):
                        try:
                            meta = json.loads(t["metadata"])
                        except (json.JSONDecodeError, TypeError):
                            meta = {}
                    if not isinstance(meta, dict):
                        meta = {}
                    if meta.get("format") != "lora_raw":
                        fmt = meta.get("format") or (
                            "compressed sequence" if meta.get("compressed")
                            else "no format key")
                        skipped[fmt] = skipped.get(fmt, 0) + 1
                        continue
                    # Simple images+captions format (manager/builder.py's
                    # run_lora_ingestion_task): one clean latent, no
                    # noise/timestep baked in at all -- sampled fresh every
                    # __iter__() call (every epoch), not here; see
                    # __iter__/_materialize and manager/t_sampling.py.
                    all_samples.append({
                        "x0":          loader.get_image_latent(t["shard_index"]),
                        "prompt":      t["prompt"],
                        "neg_prompt":  meta.get("neg", ""),
                        "seed":        t["seed"],
                        "metadata":    t["metadata"],
                        "traj_type":   meta.get("type", "good"),
                    })
            finally:
                loader.close()
        if skipped:
            print(f"  [DataLoader] skipped {sum(skipped.values())} non-single-latent "
                  f"trajectory(s): {dict(skipped)} -- LoRA training reads clean "
                  "latents only; regenerate with 'LoRA (Images + Captions)' ingestion.")
        return all_samples

    def _materialize(self, s: Dict) -> Dict:
        """Turn a single-latent sample (just x0) into a trainable one --
        fresh noise every call and a timestep from the run's
        TrainTimeSampler (static/adaptive/exact -- manager/t_sampling.py),
        which is the point: called from __iter__ per batch, not from
        _load_all_samples (which is cached and would otherwise bake one
        fixed draw in for the loader's whole lifetime)."""
        x0 = s["x0"]
        try:
            model_type = json.loads(s["metadata"]).get("model_type", "eps")
        except (json.JSONDecodeError, TypeError):
            model_type = "eps"

        t_val = self._t_sampler.draw(random)
        at, st = get_alpha_sigma(t_val)
        eps = torch.randn_like(x0)
        x_t = x0 + st * eps
        target = eps_to_vpred(eps, x_t, at, st) if model_type == "vpred" else eps

        out = dict(s)
        out["x_t"] = x_t
        out["target"] = target
        out["target_p"] = None
        out["target_n"] = None
        out["t"] = t_val
        return out

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
            size = s["x0"].shape[2:]
            key = (s["prompt"], s["neg_prompt"], size)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(s)

        if (self.shuffle and not self.keep_incomplete and self.batch_size > 1
                and not self._warned_dropping):
            self._warned_dropping = True
            never = sum(len(v) for v in buckets.values() if len(v) < self.batch_size)
            partial = sum(len(v) % self.batch_size for v in buckets.values()
                          if len(v) >= self.batch_size)
            if never or partial:
                total = len(self._samples)
                print(f"  [DataLoader] WARNING: batch_size={self.batch_size} with shuffle "
                      f"drops incomplete (prompt, size) groups: {never} of {total} samples "
                      f"sit in groups smaller than a batch and are NEVER trained on, and "
                      f"{partial} more are skipped (a random subset) each epoch -- only "
                      f"{total - never - partial} of {total} are used per epoch. Set "
                      f"keep_incomplete_batches=True to train on all of them (as smaller "
                      f"batches), or use batch_size=1.")

        # 2. Shuffle within buckets and form batches
        all_batches = []
        for key, samples in buckets.items():
            if self.shuffle:
                random.shuffle(samples)
            
            for i in range(0, len(samples), self.batch_size):
                chunk = samples[i : i + self.batch_size]
                # Drop incomplete last batch if shuffling (common training practice)
                if len(chunk) < self.batch_size and self.shuffle and not self.keep_incomplete:
                    continue
                # Fresh noise + TrainTimeSampler t per sample, every batch
                # (see _materialize).
                chunk = [self._materialize(s) for s in chunk]
                all_batches.append(self._merge_samples(chunk))

        if not all_batches:
            if self._samples:
                # Real bug this used to be, not a defensive rewrite:
                # dropping every bucket's incomplete last chunk when
                # shuffling (the branch just above) is correct and
                # intentional in general -- but if EVERY bucket's samples
                # fit in one incomplete chunk (the whole dataset is
                # smaller than batch_size, or every per-(prompt,size)
                # bucket is), dropping "the last incomplete chunk" drops
                # *all* of it, every single epoch, forever. __iter__()
                # used to just silently `return` here -- an empty
                # generator, not an error -- which surfaces several
                # frames away and looking unrelated: FetchBatchPhase
                # (nodes/train/step_pipeline.py) catches exactly one
                # StopIteration to wrap to a new epoch, gets a second,
                # uncaught StopIteration immediately after (the fresh
                # epoch is just as empty), and that crashes the whole
                # training run with a bare "StopIteration" and no
                # indication why. Reproduced directly with a real 1-row
                # sqlite dataset (shuffle=True, batch_size=2): len(loader)
                # reports 1 (misleading -- see __len__'s own note below),
                # list(loader) yields zero batches. Raising here instead
                # turns that into an immediate, specific, first-step
                # error instead of a confusing crash one retry later.
                raise ValueError(
                    f"This dataset has {len(self._samples)} sample(s) after bucketing "
                    f"by (prompt, neg_prompt, size), but batch_size={self.batch_size} "
                    f"and shuffle=True -- every bucket's samples fit in one incomplete "
                    f"chunk, and an incomplete chunk is dropped when shuffling (common "
                    f"training practice, see this method's own comment above), so no "
                    f"batch can ever be formed. Reduce batch_size, add more samples to "
                    f"this dataset, or set shuffle=False."
                )
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
        """Return the actual number of samples that iteration will yield.

        Note: this is NOT simply sum(sample_count) over trajectories —
        _load_all_samples() drops every sample where t == 0 (see comment
        there), so the raw DB sample_count overcounts whenever any
        trajectory contains a t=0 sample. To stay accurate we materialize
        (and cache) the sample list here if it isn't already loaded; this
        is the same cache __iter__/_load_all_samples populate, so calling
        __len__ before iterating does not cause a second disk read.
        """
        if self._samples is None:
            self._samples = self._load_all_samples()
        return len(self._samples)
