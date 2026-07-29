"""RenoiseBatchSource: decorates any TrainingBatchSource, replacing each
batch's precomputed (x_t, target, t) with an independently, freshly
resampled noise/timestep draw.

Why this exists: manager/builder.py's run_ingestion_task (the real-image
ingestion path) builds one timestep grid -- t_grid, ~20 values by default
-- ONCE, before the per-image loop, and reuses that *exact same* grid for
every image in the dataset:

    rng = random.Random(seed)
    if t_mode == "uniform":
        ... t_grid = [...]                      # built once
    for i, img_path in enumerate(...):           # per image
        for t_val in t_grid:                     # same grid every time
            ...

So every sample in a dataset built this way sits at one of only ~20
exact timestep values (18 of them jittered, but by the *same* fixed
draws, and the two endpoints t_low/t_high completely unjittered),
repeated identically across the entire dataset and every epoch. That's a
strong, artificial regularity mainstream training doesn't have -- kohya/
diffusers/OneTrainer all resample a fresh, continuous timestep per image
every time it's used, precisely to avoid a model latching onto anything
timestep-grid-shaped rather than learning a smooth function of noise
level. A LoRA adapter (small, low-rank, easy to overfit) has a
plausible, direct path to learning a shortcut keyed to those exact
timestep embeddings instead of the general denoising function --
consistent with the coarse/destructive-early-change symptom this project
has been fighting.

Fixing this at the source means re-ingesting (expensive: full VAE
re-encode of every image). This node avoids that: x0 (the clean latent)
is exactly recoverable from what's already stored, because ingestion's
own forward process is invertible and this project already has the
inverse functions for it (core/noise_schedule.py's eps_to_x0/vpred_to_x0
-- reused via composition, not reimplemented):

    ingestion:  x_t = x0 + sigma * eps,  target = eps (or eps_to_vpred(eps, ...))
    recovery:   x0 = eps_to_x0(target, x_t, alpha, sigma)   [eps models]
                x0 = vpred_to_x0(target, x_t, alpha, sigma) [vpred models]

alpha/sigma for the *stored* t comes from core.noise_schedule.get_alpha_sigma(t)
-- the exact same function manager/builder.py's ingestion itself calls to
build at_f/st_f, so this is an exact inversion, not an approximation
(verified against ingestion's real formula, not assumed). model_type
comes from the batch's own "metadata" JSON string (same field, same
json.loads(...)["model_type"] convention manager/loader.py already uses,
defaulting to "eps" the same way it does).

Once x0 is recovered, a fresh continuous timestep and fresh noise are
drawn independently per sample in the batch (not one shared draw for the
whole batch), and the forward process is reapplied at the new timestep --
same formula ingestion used, so the resulting (x_t, target) pair is
exactly as valid as one ingestion would have produced at that timestep,
just never actually baked into a file.

Limitation, stated plainly: target_p/target_n (the dual-CFG-pass fields)
are only regenerated correctly for the real-image ingestion path, where
manager/builder.py's own code sets target_p == target_n == target by
construction (verified directly in that file, not assumed) -- there's no
teacher model here to regenerate genuinely distinct positive/negative
targets from. A dataset built from actual teacher-trajectory distillation
(different code path, target_p != target_n) would get incorrectly
collapsed by this decorator; don't use it on that kind of dataset.
"""

from __future__ import annotations

import json
import random
from typing import ClassVar, Iterator, Optional

from ..core import Port
from .handle import TrainingBatchSource
from .node import DataSourceNode


class RenoiseBatchSource(TrainingBatchSource):

    def __init__(self, inner: TrainingBatchSource, t_low: int = 1, t_high: int = 999,
                 t_mode: str = "uniform", seed: Optional[int] = None):
        self._inner = inner
        self._t_low = t_low
        self._t_high = t_high
        self._t_mode = t_mode
        self._rng = random.Random(seed)
        import torch
        self._torch_gen = torch.Generator()
        if seed is not None:
            self._torch_gen.manual_seed(seed)

    def __iter__(self) -> Iterator[dict]:
        for batch in self._inner:
            yield self._renoise(batch)

    def __len__(self) -> int:
        return len(self._inner)

    def invalidate(self) -> None:
        self._inner.invalidate()

    def _renoise(self, batch: dict) -> dict:
        import torch

        from core.noise_schedule import eps_to_vpred, eps_to_x0, get_alpha_sigma, sample_timestep, vpred_to_x0

        x_t = batch["x_t"]
        target = batch["target"]
        t_orig = batch["t"]
        batch_size = x_t.shape[0]

        try:
            model_type = json.loads(batch.get("metadata") or "{}").get("model_type", "eps")
        except (json.JSONDecodeError, TypeError):
            model_type = "eps"

        at_orig, st_orig = get_alpha_sigma(t_orig)
        at_orig = at_orig.view(-1, 1, 1, 1).float()
        st_orig = st_orig.view(-1, 1, 1, 1).float()
        if model_type == "vpred":
            x0 = vpred_to_x0(target.float(), x_t.float(), at_orig, st_orig)
        else:
            x0 = eps_to_x0(target.float(), x_t.float(), at_orig, st_orig)

        t_new = torch.tensor(
            [sample_timestep(self._rng, self._t_mode, self._t_low, self._t_high)
             for _ in range(batch_size)],
            dtype=torch.long,
        )
        at_new, st_new = get_alpha_sigma(t_new)
        at_new_b = at_new.view(-1, 1, 1, 1).float()
        st_new_b = st_new.view(-1, 1, 1, 1).float()

        eps_new = torch.randn(x0.shape, generator=self._torch_gen, dtype=torch.float32)
        x_t_new = (x0 + st_new_b * eps_new).to(x_t.dtype)
        if model_type == "vpred":
            target_new = eps_to_vpred(eps_new, x_t_new.float(), at_new_b, st_new_b).to(target.dtype)
        else:
            target_new = eps_new.to(target.dtype)

        out = dict(batch)
        out["x_t"] = x_t_new
        out["target"] = target_new
        out["t"] = t_new
        # Real-image samples only (see module docstring) -- collapsed to
        # match target_new, mirroring ingestion's own target_p==target_n
        # invariant for this path.
        if "target_p" in batch and batch["target_p"] is not None:
            out["target_p"] = target_new
        if "target_n" in batch and batch["target_n"] is not None:
            out["target_n"] = target_new
        return out


class RenoiseBatchSourceNode(DataSourceNode):
    """Wraps a TrainingBatchSource so every batch gets a freshly resampled
    timestep/noise draw instead of ingestion's fixed, shared grid. See
    this module's docstring for why -- wire this between your dataset
    source and the trainer with no other changes needed."""

    INPUTS: ClassVar[dict[str, Port]] = {
        "batches": Port(name="batches", type=TrainingBatchSource, required=True),
        "t_low": Port(name="t_low", type=int, required=False, default=1,
                      doc="Lower bound of the resampled timestep range (inclusive). "
                          "t=0 isn't valid (sigma≈0 produces unreliable targets)."),
        "t_high": Port(name="t_high", type=int, required=False, default=999,
                       doc="Upper bound of the resampled timestep range (inclusive)."),
        "t_mode": Port(name="t_mode", type=str, required=False, default="uniform",
                       doc="uniform / low / mid / high / logit -- same distributions "
                           "core.noise_schedule.sample_timestep already implements."),
        "seed": Port(name="seed", type=int, required=False, default=None,
                     doc="None = nondeterministic (fresh draws every run)."),
    }

    def build(self, **inputs) -> dict[str, TrainingBatchSource]:
        self.validate_inputs(inputs)
        result = {"batches": RenoiseBatchSource(
            inputs["batches"],
            t_low=inputs.get("t_low", self.INPUTS["t_low"].default),
            t_high=inputs.get("t_high", self.INPUTS["t_high"].default),
            t_mode=inputs.get("t_mode", self.INPUTS["t_mode"].default),
            seed=inputs.get("seed", self.INPUTS["seed"].default),
        )}
        self.validate_outputs(result)
        return result
