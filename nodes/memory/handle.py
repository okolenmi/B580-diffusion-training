"""DeviceResident: the runtime lifecycle contract shared by anything that
holds device memory as part of its normal operation (optimizer, model,
text encoder, dataset prefetch buffer), regardless of domain. See
docs/design/02-foundational-ontology.md section 1.2.

Lives next to MemoryManager (nodes/memory/manager.py), not under any one
domain package, for the same reason MemoryManager itself does: nothing
about lifecycle tracking is optimizer- or model-specific.

Naming note, worth being explicit about since it's easy to misread: this
class's release() ("drop entirely, not reversible") is a different
operation from MemoryManager's release() ("mark unused, keep the
allocation for reuse -- the cheap, reversible one; MemoryManager's free()/
free_all() is the actual-drop operation). A DeviceResident implementation
that owns pooled buffers acquired from a MemoryManager calls that
manager's free()/free_all() from inside its own release(), not the
manager's release() -- see docs/design/02-foundational-ontology.md section 1.3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class DeviceResident(ABC):
    """Three lifecycle tiers, kept distinct on purpose -- collapsing them
    is exactly the mistake nodes/memory/manager.py's module docstring
    already documents once (the reset-vs-free asymmetry bug class)."""

    @abstractmethod
    def footprint_bytes(self) -> int:
        """Best-effort current device-memory usage. Best-effort, not exact
        -- an implementation wrapping third-party internals may not be
        able to account for every buffer; document what's excluded rather
        than guessing. *Device*-memory usage specifically: 0 while
        offloaded, not the byte count of whatever's now sitting in host
        RAM instead -- numel()*element_size() alone can't tell CPU-
        resident from device-resident (shape and dtype don't change when
        a tensor moves), so every implementation has to check this
        explicitly, typically by tracking its own offloaded/not-offloaded
        state (see nodes/model/text_encoder.py or
        nodes/optimizer/composed.py for two different ways to do that).
        Got this wrong project-wide until it didn't -- see
        docs/known-issues/resolved.md."""

    @abstractmethod
    def offload(self) -> None:
        """Move to host memory. The object stays alive and identity-stable
        (same Python object, same optimizer momentum, same cache contents)
        -- this is the cheap, common, reversible operation."""

    @abstractmethod
    def reload(self, device: str | None = None) -> None:
        """Move back to device. None = wherever it was before offload()."""

    @abstractmethod
    def release(self) -> None:
        """Drop device (and possibly host) state entirely. Not reversible
        via reload() -- whatever built this object has to build it again.
        Used when a run is actually discarding something, not pausing it."""


def sum_tensor_bytes(*tensor_lists) -> int:
    """sum(t.numel() * t.element_size()) over every real tensor across any
    number of iterables, skipping None entries. Pulled out here because
    it's the same small loop every legacy-optimizer-wrapping Handle below
    needs for footprint_bytes() -- several of those wrapped optimizers
    (core.optimizers.ChunkedXPUAdafactor/ChunkedXPUCAME/ForeachXPUAdafactor/
    ForeachXPUCAME/FusedXPUAdafactor) hold their per-parameter state as
    lists of Optional[Tensor] (None until that parameter's state is
    lazily allocated on its first real step), confirmed by reading their
    __init__ methods directly rather than assumed."""
    total = 0
    for tensors in tensor_lists:
        for t in tensors:
            if t is not None:
                total += t.numel() * t.element_size()
    return total
