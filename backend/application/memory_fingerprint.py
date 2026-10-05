"""GraphMemoryFingerprint -- what makes two runs "the same configuration"
for peak purposes, computed without importing torch.

The server needs a fingerprint **before spawning** a child, and must not
import torch to get it. So the fingerprint is derived from:

1. **Declared fields** on nodes that matter (a class-level
   ``memory_fields: tuple[str, ...]`` -- names of the params that change
   the peak). This is data, not code, so the server can read it without
   instantiating anything.
2. **Dataset stats** -- latent h/w from the *largest bucket* of the
   dataset (the peak is set by the largest shape).

If any required field cannot be found the fingerprint is **unknown**,
not defaulted. An unknown fingerprint is not a zero peak; it means
nothing has been measured for this configuration.

The fields are deliberately narrow: model, batch_size, latent_h, latent_w,
rank, checkpointing, optimizer. A caption or a checkpoint path does not
change the peak; a batch size or a resolution does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain.graph import GraphDefinition


#: Fields that make two runs "the same configuration" for peak purposes.
#: Deliberately *not* a hash of the whole graph: two graphs that differ in
#: a caption or a checkpoint path but train the same shapes at the same
#: batch have the same peak, and folding that difference in would mean
#: re-measuring for every dataset shuffle.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "model",
    "batch_size",
    "latent_h",
    "latent_w",
    "rank",
    "checkpointing",
    "optimizer",
)


@dataclass(frozen=True, slots=True)
class GraphMemoryFingerprint:
    """What makes two runs "the same configuration" for peak purposes.

    Only the things that change peak. Deliberately *not* a hash of the whole
    graph: two graphs that differ in a caption or a checkpoint path but train
    the same shapes at the same batch have the same peak, and folding that
    difference in would mean re-measuring for every dataset shuffle.
    """

    model: str
    batch_size: int
    latent_h: int
    latent_w: int
    rank: int
    checkpointing: bool
    optimizer: str

    def key(self) -> str:
        """Stable string key for the peak store."""
        return "|".join(str(getattr(self, f)) for f in FINGERPRINT_FIELDS)


@dataclass(frozen=True, slots=True)
class UnknownFingerprint:
    """Sentinel for a graph whose fingerprint cannot be computed.

    Not a zero peak. An unknown fingerprint means nothing has been measured
    for this configuration, and under `check_comfy_conflicts`' rule it
    blocks admission rather than admitting a run whose peak is nominally
    nothing.
    """

    reason: str


#: A resolver takes a class name and returns the declared ``memory_fields``
#: tuple, or None if the class is unknown. Injected so this module stays
#: pure (no infrastructure imports) and testable.
MemoryFieldsResolver = Callable[[str], tuple[str, ...] | None]


def _as_int(value: object) -> int:
    """Coerce a widget value to ``int``.

    JSON params arrive as int, float or a numeric string; ``int()`` on
    the raw ``object`` is a mypy call-overload error, and coercing
    through ``str``/``float`` accepts all three shapes. Anything else is
    malformed and raises, as bare ``int()`` always did here.
    """
    return int(float(str(value)))


def graph_fingerprint(
    graph: GraphDefinition,
    dataset_stats: dict | None,
    *,
    resolve_memory_fields: MemoryFieldsResolver,
) -> GraphMemoryFingerprint | UnknownFingerprint:
    """Compute the fingerprint for a graph + dataset, without importing torch.

    `dataset_stats` is a dict with a ``buckets`` key: a list of
    ``{"height": int, "width": int, "count": int}`` dicts. The latent h/w
    come from the **largest bucket** (the peak is set by the largest shape).

    `resolve_memory_fields` is injected so this module stays pure and
    testable. The real implementation lives in the infrastructure layer
    (it needs to load node classes); tests pass a fake.

    Returns `UnknownFingerprint` if any required field cannot be found.
    """
    # -- collect declared fields from nodes that declare them ---------------
    declared: dict[str, object] = {}
    for node in graph.nodes:
        fields = resolve_memory_fields(node.class_name)
        if not fields:
            continue
        for field_name in fields:
            if field_name in node.params:
                declared[field_name] = node.params[field_name]

    # -- latent h/w from the largest dataset bucket --------------------------
    # These come from the dataset, not from node params, so they are handled
    # separately from the declared fields above.
    if dataset_stats is None:
        return UnknownFingerprint(reason="no dataset stats provided")

    buckets = dataset_stats.get("buckets")
    if not buckets:
        return UnknownFingerprint(reason="dataset stats have no buckets")

    # The peak is set by the largest shape. "Largest" means the bucket with
    # the most pixels (height * width), not the one with the most samples.
    largest = max(buckets, key=lambda b: b["height"] * b["width"])
    # Shapeless rows (latent_h/w never filled in) are unknown, not a
    # 0x0 configuration: a key that ignores shape would file this
    # dataset's peak under a key a genuinely large-shape run also
    # computes, and the read-back would then be a peak measured at the
    # wrong size.
    if largest["height"] <= 0 or largest["width"] <= 0:
        return UnknownFingerprint(
            reason="largest bucket carries no latent shape"
        )
    latent_h = largest["height"]
    latent_w = largest["width"]

    # -- required fields from node params -------------------------------------
    # latent_h and latent_w are NOT here -- they come from dataset stats above.
    node_fields = tuple(f for f in FINGERPRINT_FIELDS
                        if f not in ("latent_h", "latent_w"))
    missing: list[str] = []
    for field_name in node_fields:
        if field_name not in declared:
            missing.append(field_name)

    if missing:
        return UnknownFingerprint(
            reason=f"missing fingerprint fields: {', '.join(missing)}"
        )

    return GraphMemoryFingerprint(
        model=str(declared["model"]),
        batch_size=_as_int(declared["batch_size"]),
        latent_h=latent_h,
        latent_w=latent_w,
        rank=_as_int(declared["rank"]),
        checkpointing=bool(declared["checkpointing"]),
        optimizer=str(declared["optimizer"]),
    )
