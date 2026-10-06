"""Streaming dataset loader — reads trajectories from virtual sets."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Union
import torch

from .db import get_training_set_trajectories, get_training_set_by_name, ensure_v2
from .storage import ShardLoader
from .t_sampling import TrainTimeSampler
from nodes.components.noise_schedule import eps_to_vpred, get_alpha_sigma


class ManagedDatasetLoader:
    """Streams data from a virtual training set, with optional sample batching."""

    def __init__(self, dataset_root: Path, set_identifier: Optional[Union[int, str]] = None,
                 shuffle: bool = True, batch_size: int = 1, use_dataset_cfg: bool = True,
                 t_low: int = 1, t_high: int = 999, t_mode: str = "uniform",
                 bucket_balance=None, t_values: str = "",
                 keep_incomplete: bool = False,
                 shape_bucket_multiple: int = 0):
        self.root = dataset_root
        self.db_path = dataset_root / "metadata.db"
        self.shuffle = shuffle
        self.batch_size = batch_size
        # 0 = off (default, and what every graph built before this knob
        # existed gets). N > 1 pads each latent up to the next multiple of N so
        # a multi-resolution dataset trains on few shapes. Opt-in because it
        # changes what the loss is computed over and the order samples arrive
        # in: the padded region is excluded by a mask (LossPhase), and samples
        # are regrouped by bucketed size. See _bucket_size/_apply_bucket.
        self.shape_bucket_multiple = int(shape_bucket_multiple or 0)
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

        # After TrainTimeSampler's config validation on purpose (its errors
        # must arrive before any DB access -- pinned by smoke tests), but
        # before any query: v2 queries reference columns v1 doesn't have, and
        # a migration-guidance error beats "no such column: neg_prompt".
        #
        # A missing metadata.db means there is no dataset here, and saying so
        # is the whole point: carrying on would let the next query's
        # sqlite3.connect *create* an empty database in the directory, so
        # the failure would surface a step later as "no such table:
        # trajectories" -- naming a schema problem for what is a missing
        # dataset, and leaving the directory it made behind. Found by
        # passing a dataset name that does not exist: it created
        # datasets/<name>/ and then reported a missing table.
        #
        # Same rule as the backend's own library, which already treats a
        # directory without a metadata.db as not-a-dataset.
        if not self.db_path.exists():
            raise ValueError(
                f"No dataset at '{dataset_root}': no {self.db_path.name} in it. "
                f"Check the name against the datasets directory."
            )
        ensure_v2(self.db_path)

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
        
        # Group by shard to minimize file openings
        self.shard_map = {}
        for t in self.trajectories:
            path = t["file_path"]
            if path not in self.shard_map:
                self.shard_map[path] = []
            self.shard_map[path].append(t)

    def _load_all_samples(self) -> list:
        """Load every single-latent sample into a flat list.

        One entry per image -- ingestion stores one clean latent, so there
        is no per-timestep structure left to interleave (the docstring this
        replaced described the retired baked-grid format). Shards whose
        layout is not "single_latent" (teacher/compressed sequences, or any
        legacy/unknown layout) are skipped as a whole, not guessed at: they
        are not single-latent data, and reading them as such would be wrong.
        The list is held in RAM; for typical datasets (100-1000 images) this
        is well under 1 GB.
        """
        all_samples = []
        skipped = {}
        for path, trajs in self.shard_map.items():
            # All rows of one shard file share its layout (one file, one key
            # layout -- see docs/design/backend/04-dataset-format.md).
            layout = trajs[0].get("layout") or "unknown"
            if layout != "single_latent":
                skipped[layout] = skipped.get(layout, 0) + len(trajs)
                continue
            loader = ShardLoader(self.root / path)
            loader.load()
            try:
                for t in trajs:
                    # Simple images+captions format (manager/builder.py's
                    # run_lora_ingestion_task): one clean latent, no
                    # noise/timestep baked in at all -- sampled fresh every
                    # __iter__() call (every epoch), not here; see
                    # __iter__/_materialize and manager/t_sampling.py.
                    neg = t.get("neg_prompt") or ""
                    model_type = t.get("model_type") or "eps"
                    traj_type = t.get("type") or "good"
                    all_samples.append({
                        "x0":          loader.get_image_latent(t["shard_index"]),
                        "prompt":      t["prompt"],
                        "neg_prompt":  neg,
                        "seed":        t["seed"],
                        "model_type":  model_type,
                        "traj_type":   traj_type,
                        # v2 stores these as columns; the key survives for
                        # batch-dict contract stability (no consumer parses
                        # it -- trainer reads neg_prompt directly).
                        "metadata":    json.dumps({"neg": neg, "model_type": model_type,
                                                   "type": traj_type}),
                    })
            finally:
                loader.close()
        if skipped:
            print(f"  [DataLoader] skipped {sum(skipped.values())} non-single-latent "
                  f"trajectory(s): {dict(skipped)} -- LoRA training reads clean "
                  "latents only; regenerate with 'LoRA (Images + Captions)' ingestion.")
        return all_samples

    def _bucket_size(self, h: int, w: int) -> tuple[int, int]:
        """The (h, w) this sample trains at, and whether it was bucketed.

        With ``shape_bucket_multiple`` unset (the default) this returns the
        sample's own size, so every existing graph trains on exactly the
        shapes it trained on before. That is the whole reason bucketing is a
        knob and not a behaviour change: a dataset of one shape cannot tell
        the difference, and a dataset of 63 can opt in deliberately.
        """
        m = self.shape_bucket_multiple
        if m <= 1:
            return h, w
        H = ((h + m - 1) // m) * m      # round UP: keep every pixel
        W = ((w + m - 1) // m) * m
        return H, W

    def _apply_bucket(self, x0: torch.Tensor, H: int, W: int
                      ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pad (C, h, w) up to (C, H, W); return it with a (H, W) validity mask.

        Returns None when the sample is already the bucket size, so the common
        "no padding needed" path allocates nothing and the batch carries no
        mask at all.

        **Rounds up, never down.** Rounding down would be cheaper in compute
        and need no mask, but it discards real pixels: on `non-square`,
        cropping to a multiple of 64 reaches a single shape by throwing away
        52% of every image. Padding keeps the data and pays in compute (+15%
        at a multiple of 32, measured), which is the trade this feature exists
        to make explicit rather than to hide.

        The pad offset is drawn per sample so the model sees borders on every
        side over an epoch, rather than learning that the bottom and right are
        always real. The mask is what makes the padding safe: without it the
        loss would train on pad pixels (see LossPhase).
        """
        h, w = int(x0.shape[-2]), int(x0.shape[-1])
        if (H, W) == (h, w):
            return None
        ph, pw = H - h, W - w
        top = random.randint(0, ph) if ph else 0
        left = random.randint(0, pw) if pw else 0
        # Shape-generic over the leading dims: a sample's latent is (C, h, w)
        # in some loaders and (1, C, h, w) in others, and this must not care.
        out = x0.new_zeros((*x0.shape[:-2], H, W))
        valid = x0.new_zeros((*x0.shape[:-2], H, W), dtype=torch.float32)
        out[..., top:top + h, left:left + w] = x0
        valid[..., top:top + h, left:left + w] = 1.0
        return out, valid

    def _materialize(self, s: Dict) -> Dict:
        """Turn a single-latent sample (just x0) into a trainable one --
        fresh noise every call and a timestep from the run's
        TrainTimeSampler (static/adaptive/exact -- manager/t_sampling.py),
        which is the point: called from __iter__ per batch, not from
        _load_all_samples (which is cached and would otherwise bake one
        fixed draw in for the loader's whole lifetime)."""
        x0 = s["x0"]
        model_type = s.get("model_type") or "eps"

        # Optional shape bucketing: pad this sample up to its bucket before
        # any noise is drawn, so the noise and the target describe the same
        # padded tensor the model will see.
        valid_mask = None
        H, W = self._bucket_size(int(x0.shape[-2]), int(x0.shape[-1]))
        if self.shape_bucket_multiple > 1:
            applied = self._apply_bucket(x0, H, W)
            if applied is None:
                # Already the bucket size. Emit an all-ones mask anyway, so
                # every batch in a bucketed run carries one. A bucket can mix
                # shapes that need padding with shapes that do not (48x64 pads,
                # 64x64 does not), and a mask present only for the padded
                # members would be dropped for the whole batch by the merge --
                # training on padding for exactly the batches that have it.
                # All-ones is not a special case in the loss: sum/n == mean.
                valid_mask = x0.new_ones((*x0.shape[:-2], H, W),
                                         dtype=torch.float32)
            else:
                x0, valid_mask = applied
                s = dict(s)
                s["x0"] = x0

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
        if valid_mask is not None:
            # Only present when padding happened. Absent is the common case
            # and means "every element is real", so the loss's own code path is
            # unchanged for un-bucketed graphs.
            out["valid_mask"] = valid_mask
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
        if samples[0].get("valid_mask") is not None:
            # Present for every sample in a bucketed run (see _materialize:
            # all-ones where no padding was needed), so the whole batch has one
            # and the loss always knows which elements are real.
            out["valid_mask"] = torch.cat(
                [s["valid_mask"] for s in samples], dim=0)
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
            h, w = int(s["x0"].shape[-2]), int(s["x0"].shape[-1])
            if self.shape_bucket_multiple > 1:
                # Bucket by the shape the sample will actually train at, not
                # the shape it was stored at. Two stored sizes that round to
                # the same bucket must land in the same group, or the run pays
                # a transition it was trying to avoid.
                size = self._bucket_size(h, w)
            else:
                size = (h, w)
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
