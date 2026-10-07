"""ManagedDatasetSourceNode: wraps manager.loader.ManagedDatasetLoader.

Adapter only -- no dataset logic reimplemented here. ManagedDatasetLoader
already implements __iter__/__len__/invalidate_cache correctly (bucketing,
shard caching); this just makes it satisfy TrainingBatchSource.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Iterator

from ..core import Port
from ..components.layout import ProjectLayout
from ..train.bucket_balance import BucketBalance
from .handle import TrainingBatchSource
from .node import DataSourceNode
from .timestep_modes import T_MODES_TRAIN_TIME


class ManagedDatasetBatchSource(TrainingBatchSource):

    def __init__(self, loader):
        self._loader = loader

    def __iter__(self) -> Iterator[dict]:
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def invalidate(self) -> None:
        self._loader.invalidate_cache()


class ManagedDatasetSourceNode(DataSourceNode):
    """Streams batches from a manager-managed dataset (safetensors shards + sqlite index)."""

    #: Fields that change the peak VRAM this node's run will reach. Declarative
    #: data (not code) so the server can compute a graph fingerprint without
    #: importing torch or instantiating anything. See
    #: backend/application/memory_fingerprint.py.
    memory_fields: ClassVar[tuple[str, ...]] = (
        "batch_size",
    )

    INPUTS: ClassVar[dict[str, Port]] = {
        "dataset_root": Port(name="dataset_root", type=Path, required=True, path_kind="dataset",
                              doc="Dataset name from the library. Absolute paths and '..' are rejected -- "
                                  "this field is reachable from the graph editor over the network, so "
                                  "it's sandboxed to the configured datasets directory."),
        "set_identifier": Port(name="set_identifier", type=Any, required=False, default=None,
                                doc="Training-set name or ID; None = every trajectory in the dataset."),
        "shuffle": Port(name="shuffle", type=bool, required=False, default=True),
        "batch_size": Port(name="batch_size", type=int, required=False, default=1),
        "use_dataset_cfg": Port(name="use_dataset_cfg", type=bool, required=False, default=True,
                                doc="Legacy, now a no-op: it only ever gated the retired "
                                    "baked-format path's dual-pass target blending. "
                                    "Kept so existing graphs still load unchanged."),
        "t_low": Port(name="t_low", type=int, required=False, default=1,
                      doc="Low end of the t range each sample's noise timestep is drawn "
                          "from at train time (ingestion stores clean latents only, so "
                          "t is chosen per draw -- see manager/t_sampling.py)."),
        "t_high": Port(name="t_high", type=int, required=False, default=999,
                       doc="High end of that range (inclusive)."),
        "t_mode": Port(name="t_mode", type=str, required=False, default="uniform",
                       choices=T_MODES_TRAIN_TIME,
                       doc="How t is drawn for each sample at train time: the five "
                           "static distributions core.noise_schedule.sample_timestep "
                           "implements, plus two train-time modes -- 'adaptive': "
                           "bucket ~ (current/baseline)^sample_bias over the buckets "
                           "[t_low, t_high] actually covers, read from a wired Bucket "
                           "Balance node (requires the bucket_balance input); "
                           "'exact': t pinned to t_values -- cycled one value per "
                           "sample drawn, or steered across that list by a wired "
                           "bucket_balance (each listed t's bucket difficulty picks "
                           "which value gets hit; without one it stays a plain "
                           "cycle). Anything else fails at build time rather than "
                           "degrading silently to uniform."),
        "t_values": Port(name="t_values", type=str, required=False, default="",
                         doc="t_mode='exact' only: comma-separated timesteps to pin t "
                             "to, e.g. '500' (every sample at t=500) or '200,500,800' "
                             "(cycled in draw order -- equal long-run share per value "
                             "regardless of shuffling). Wire bucket_balance to steer "
                             "that list toward the buckets the balance measures as "
                             "behind instead of cycling uniformly. Every value must "
                             "be an integer inside [t_low, t_high] (and 1..999); "
                             "ignored by other modes.",
                         visible_when=("t_mode", "exact")),
        "bucket_balance": Port(
            name="bucket_balance", type=BucketBalance, required=False, default=None,
            doc="Required when t_mode='adaptive' -- the same Bucket Balance instance "
                "the trainer is wired to (its observe() is what the sampling reads; "
                "without the trainer side it would track nothing and stay uniform). "
                "Optional with t_mode='exact': wired, the balance steers draws across "
                "t_values toward the buckets it measures as behind; unwired (or with "
                "nothing to report yet), t_values stays the plain pinned cycle. "
                "None with any static t_mode = today's behavior, unchanged.",
            visible_when=("t_mode", ("adaptive", "exact"))),
        "shape_bucket_multiple": Port(
            name="shape_bucket_multiple", type=int, required=False, default=0,
            doc="0 (default, off) trains on the shapes the dataset stores. "
                "N > 1 pads each latent up to the next multiple of N, so a "
                "multi-resolution dataset trains on few shapes. Measured on "
                "`non-square` at batch 2: a multiple of 32 collapses 44 shapes "
                "to 3, cuts first-sighting compiles from 159 s to 12 s, and "
                "runs 1.83x faster over 150 steps (0.599 -> 1.096 steps/s). "
                "The permanent cost is small: +1% per step (+0.010 s), well "
                "under the +15% a latent-pixel count suggests, because compute "
                "here is not proportional to pixel count. It also trains "
                "273/273 samples instead of 242/273, since grouping by "
                "bucketed size empties the incomplete (caption, size) groups "
                "that were being dropped. Off by default because it changes "
                "which pixels the loss covers and the order samples arrive in. "
                "Padded elements are excluded from the loss by a mask -- so "
                "nothing is trained on padding, and the loss is not rescaled "
                "by the padded fraction -- and the pad offset is drawn per "
                "sample so borders do not become systematically real. Pads up, "
                "never down: cropping to a multiple of 64 would reach a single "
                "shape by discarding 52% of every image.",
        ),
        "keep_incomplete_batches": Port(
            name="keep_incomplete_batches", type=bool, required=False, default=False,
            doc="Batches are formed per identical (caption, image size) group. With "
                "shuffle on, an incomplete last batch of each group is dropped -- so with "
                "per-image captions and batch_size > 1, any image whose caption is unique "
                "(or whose group is smaller than batch_size) is never trained on, silently. "
                "False = that historical behavior (a warning with the exact counts is "
                "printed at the first epoch). True = keep those samples as smaller batches "
                "so every image is trained every epoch; the cost is extra batch shapes "
                "(one more per resolution), which can add a one-time kernel-warmup stall."),
        "project_layout": Port(
            name="project_layout", type=ProjectLayout, required=False, default=None,
            doc="None = ProjectLayout.from_paths_module() -- see nodes/components/layout.py.",
        ),
    }

    def build(self, **inputs) -> dict[str, TrainingBatchSource]:
        self.validate_inputs(inputs)
        from manager.loader import ManagedDatasetLoader

        # Before any path resolution: t misconfiguration is a config error
        # and should fail as one, not as a loader crash mid-iteration (the
        # loader itself repeats every one of these checks -- it's a public
        # constructor -- but only after this node would already have
        # resolved paths).
        t_mode = inputs.get("t_mode", self.INPUTS["t_mode"].default)
        bucket_balance = inputs.get("bucket_balance")
        t_low = inputs.get("t_low", self.INPUTS["t_low"].default)
        t_high = inputs.get("t_high", self.INPUTS["t_high"].default)
        t_values = inputs.get("t_values", self.INPUTS["t_values"].default)
        if t_mode == "adaptive" and bucket_balance is None:
            raise ValueError(
                "ManagedDatasetSourceNode: t_mode='adaptive' requires the "
                "bucket_balance input -- wire a Bucket Balance node's output (it is "
                "what knows per-bucket progress to steer sampling by). Any static "
                "t_mode works without it.")
        if t_mode == "exact":
            # Parse/validate the cycled list here for the same reason:
            # a typo'd t_values is a config error, raised before any fs/DB work.
            from manager.t_sampling import parse_exact_t_values
            parse_exact_t_values(t_values, t_low, t_high)

        layout = inputs.get("project_layout") or ProjectLayout.from_paths_module()
        loader = ManagedDatasetLoader(
            dataset_root=layout.resolve_safe_dataset_path(str(inputs["dataset_root"])),
            set_identifier=inputs.get("set_identifier", self.INPUTS["set_identifier"].default),
            shuffle=inputs.get("shuffle", self.INPUTS["shuffle"].default),
            batch_size=inputs.get("batch_size", self.INPUTS["batch_size"].default),
            use_dataset_cfg=inputs.get("use_dataset_cfg", self.INPUTS["use_dataset_cfg"].default),
            t_low=t_low,
            t_high=t_high,
            t_mode=t_mode,
            bucket_balance=bucket_balance,
            t_values=t_values,
            keep_incomplete=inputs.get(
                "keep_incomplete_batches", self.INPUTS["keep_incomplete_batches"].default),
            shape_bucket_multiple=inputs.get(
                "shape_bucket_multiple", self.INPUTS["shape_bucket_multiple"].default),
        )
        # Size the oneDNN primitive cache to this dataset's shape count
        # (nodes/xpu_env.py's own docstring for why it belongs here).
        #
        # This node is the trigger, and the reason is graph execution order:
        # nodes build in topological order (backend/infrastructure/graph/
        # runtime.py), so the dataset is built before the trainer that consumes
        # its batches. It is also the only node that knows how many distinct
        # latent shapes the run will see.
        #
        # Best-effort by construction: the sizing must never be the reason a
        # dataset fails to load, so anything unexpected here is swallowed and
        # the default capacity stands. That default is measured to cover
        # `non-square`, and capacity is a ceiling rather than an allocation,
        # so a run that skips this step still works -- it is just sized for
        # one dataset rather than its own.
        try:
            shapes = {
                (row.get("latent_h"), row.get("latent_w"))
                for row in (loader.trajectories or [])
                if row.get("latent_h") and row.get("latent_w")
            }
            # Over-counts on purpose (63 rows -> 44 trained shapes for
            # non-square); the module docstring says why that is the safe
            # direction.
            from ..xpu_env import apply_primitive_cache_capacity_for_shapes
            apply_primitive_cache_capacity_for_shapes(len(shapes))
        except Exception as exc:  # noqa: BLE001 -- sizing is never load's business
            import logging
            logging.getLogger(__name__).warning(
                "could not size the oneDNN primitive cache to this dataset "
                "(%s: %s); keeping the default capacity, which is measured to "
                "cover a 44-shape dataset", type(exc).__name__, exc,
            )

        # What bucketing costs this dataset, once, before any step runs: the
        # pad fraction per sample and its distribution. Reported here because
        # build() is the only point that knows the dataset's shapes without
        # having iterated it, and because the permanent cost of the knob is
        # invisible otherwise -- the one-time saving is in the speed, the
        # recurring cost is in the padding, and only one of them shows up in
        # a step time.
        #
        # A no-op when bucketing is off, which is the default: there is no
        # padding to report and the report would only be noise.
        try:
            loader.report_pad_fraction()
        except Exception as exc:  # noqa: BLE001 -- a report is never load's business
            import logging
            logging.getLogger(__name__).warning(
                "could not report bucketing's pad fraction for this dataset "
                "(%s: %s); training is unaffected either way", type(exc).__name__,
                exc,
            )

        result = {"batches": ManagedDatasetBatchSource(loader)}
        self.validate_outputs(result)
        return result
