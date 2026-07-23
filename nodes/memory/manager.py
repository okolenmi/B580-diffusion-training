"""MemoryManager: centralized, explicit device-buffer lifecycle.

Motivation, stated precisely rather than "because centralizing memory is
good practice" -- strategies/chunked.py's own module docstring already
flagged two concrete gaps in its first version: the scratch buffer was
allocated fresh on every step() call (no cross-step caching), and there
was no torch.xpu.MemPool integration. Building a second ad hoc
`self._scratch` attribute directly on that strategy to fix the first gap
would have been exactly the kind of thing this package's design already
rejected once: `docs/nodes_package_design.md`'s "Course correction"
section documents a real reset-vs-free asymmetry bug
(`_scratch`/`_pool` cleared in `free_states()` but not `reset_states()`
in some of `core/optimizers.py`'s legacy classes) that came directly from
several classes each hand-managing their own scratch-buffer attributes.
That bug class isn't specific to the old code's shape -- any class that
owns a cached buffer and exposes multiple lifecycle hooks (offload, free,
reset) that each need to remember to touch it is one forgotten line away
from the same mistake. Centralizing acquire/release/free through one
reviewed class, used by anything that needs a reusable buffer, means
there's exactly one place this can go wrong, not N ad hoc copies of it.

Deliberately a plain, explicit, injectable object -- NOT a global
singleton or module-level dict. Anything that wants shared buffers (e.g.
multiple ExecutionStrategy instances working on the same OptimizerHandle,
or -- looking ahead -- a future non-optimizer node domain, such as the
preview-generation VRAM growth tracked in docs/suspicious_findings.md)
constructs one instance and passes it around, exactly like Algorithm and
ExecutionStrategy are already passed around explicitly rather than looked
up implicitly. Domain-independent on purpose: lives next to nodes/core.py,
not under nodes/optimizer/, so nothing about it is optimizer-specific.

Three distinct operations, kept separate on purpose (collapsing them into
one method is exactly the kind of shortcut that produces the asymmetry
bug class described above):
  - get_buffer(): acquire-or-reuse a named buffer, marks it in-use.
  - release(): mark a buffer no longer needed *this call*, but keep the
    underlying allocation around for next time -- the cheap, common path
    between steps, and what actually gives cross-step caching.
  - free() / free_all(): actually drop the reference so the underlying
    device memory can be reclaimed -- used when a caller genuinely needs
    the memory back (e.g. before an offload round trip, or when a handle
    is being discarded), not just made available for later reuse.

What this does NOT do: no automatic eviction under memory pressure -- a
manager holds whatever it's been asked to hold until told to release/
free it, on purpose, so behavior stays predictable rather than depending
on runtime memory conditions. No cross-process or cross-device-transfer
support.

**torch.xpu.MemPool integration, added this round -- opt-in, stated
precisely rather than assumed safe.** `get_buffer()`'s single
`torch.empty()` call was always documented as the seam this would wrap
through later; `MemoryManager(use_mempool=True)` now does that, routing
every allocation for XPU-device tags through a per-device
`torch.xpu.MemPool()` via `torch.xpu.use_mem_pool()`, confirmed against
PyTorch's actual source (`torch/xpu/memory.py`) rather than assumed --
`torch.xpu.MemPool(allocator=None, use_on_oom=False)` and
`torch.xpu.use_mem_pool(pool, device=None)` are the real, current
signatures. Two real, documented tradeoffs found while researching this
(from PyTorch's own issue tracker) and worth knowing before turning it
on, not glossed over:

- Allocations inside `use_mem_pool` don't get the default caching
  allocator's normal OOM-retry-with-defragmentation behavior --
  `pytorch/pytorch#159674` reports a real OOM under `use_mem_pool` at a
  point where the default allocator would have succeeded by retrying
  after a cache flush. A MemPool trades some of that resilience for
  reduced fragmentation.
- Nesting two `use_mem_pool` context managers has a real, currently-open
  bug (`pytorch/pytorch#161193`) where the second pool silently gets no
  allocations at all. Not this class's usage pattern (each `get_buffer()`
  call enters and exits its own context around a single `torch.empty()`,
  never nested), but worth knowing if this class is ever extended.

Default `use_mempool=False` -- explicit opt-in only, so nothing about
this class's already-verified behavior changes for existing callers.
**Not yet verified on real XPU hardware by anyone** -- CPU can confirm
the plumbing (default-off path unaffected; `use_mempool=True` on a
non-XPU device raises a clear error rather than failing confusingly deep
inside `get_buffer()`) but not the actual fragmentation-reduction claim
or either risk above in practice. See
`nodes/smoke_tests/smoke_test_memory_manager.py` for what is and isn't
covered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class _BufferKey:
    """Identity of one pooled buffer. Same tag requested with a different
    dtype or device is deliberately a *different* buffer, not an error --
    e.g. two strategies sharing one manager but running on different
    devices shouldn't have to coordinate tag names to avoid colliding."""
    tag: str
    dtype: torch.dtype
    device: str


@dataclass
class _Buffer:
    tensor: torch.Tensor
    in_use: bool = False


class MemoryManager:
    """Owns a set of named, reusable device buffers ("tags"), each grown
    lazily to the largest size ever requested for that tag and kept alive
    across get_buffer() calls until explicitly release()d (logically
    freed, allocation kept) or free()d (allocation actually dropped).
    See module docstring for the full reasoning, including the
    use_mempool option's real, documented tradeoffs.
    """

    def __init__(self, use_mempool: bool = False) -> None:
        self._buffers: dict[_BufferKey, _Buffer] = {}
        self._use_mempool = use_mempool
        self._mempools: dict[str, Any] = {}  # device str -> torch.xpu.MemPool
        if use_mempool and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError(
                "MemoryManager(use_mempool=True) requires a working torch.xpu "
                "backend (torch.xpu.is_available() must be True). A build can "
                "expose the torch.xpu.MemPool class while it's still a "
                "non-functional stub ('Tried to instantiate dummy base class "
                "MemPool') if no real XPU backend is compiled in -- confirmed "
                "directly in this session's own (CUDA-only) sandbox build, which "
                "is why this class checks is_available() rather than just "
                "hasattr(). Construct with use_mempool=False (the default) instead."
            )

    def _mempool_for(self, device):
        """Lazily create (or return the existing) per-device MemPool, or
        None if use_mempool is off or the device isn't XPU -- MemPool is
        XPU-specific, so a manager serving mixed devices (unusual, but
        not disallowed) only pools the XPU ones."""
        if not self._use_mempool:
            return None
        device_str = str(device)
        if not device_str.startswith("xpu"):
            return None
        if device_str not in self._mempools:
            self._mempools[device_str] = torch.xpu.MemPool()
        return self._mempools[device_str]

    def get_buffer(self, tag: str, numel: int, dtype: torch.dtype, device) -> torch.Tensor:
        """Return a 1-D tensor of at least `numel` elements registered
        under `tag` (for this dtype/device). Reuses the existing buffer
        if one is already large enough; grows (reallocates) if not;
        never shrinks, so a tag's buffer only gets reallocated as often
        as its largest-ever request changes. If use_mempool=True and
        device is XPU, a (re)allocation routes through this manager's
        per-device torch.xpu.MemPool -- see module docstring.

        Raises RuntimeError if this tag is already marked in-use (i.e.
        get_buffer() was called for it and neither release() nor free()
        has been called since) -- re-acquiring it here would silently
        alias the same storage across two live users, which is exactly
        the kind of aliasing bug docs/nodes_package_design.md notes was
        already caught and fixed once, in core/optimizers.py's
        ChunkedXPUCAME. Failing loudly here catches that class of mistake
        at the point it happens instead of as silently-wrong numbers
        somewhere downstream.
        """
        key = _BufferKey(tag=tag, dtype=dtype, device=str(device))
        buf = self._buffers.get(key)
        if buf is not None and buf.in_use:
            raise RuntimeError(
                f"MemoryManager: tag {tag!r} (dtype={dtype}, device={device}) "
                f"is already in use -- call release() or free() before "
                f"acquiring it again. Re-acquiring a live buffer would "
                f"silently alias the same storage across two callers."
            )
        if buf is None or buf.tensor.numel() < numel:
            pool = self._mempool_for(device)
            if pool is not None:
                with torch.xpu.use_mem_pool(pool, device=device):
                    tensor = torch.empty(numel, dtype=dtype, device=device)
            else:
                tensor = torch.empty(numel, dtype=dtype, device=device)
            buf = _Buffer(tensor=tensor)
            self._buffers[key] = buf
        buf.in_use = True
        return buf.tensor

    def release(self, tag: str, dtype: torch.dtype | None = None, device=None) -> None:
        """Mark buffer(s) registered under `tag` as no longer in use.
        Keeps the underlying allocation alive for the next get_buffer()
        call with the same tag -- this is the common, cheap path between
        steps. Idempotent and permissive: releasing a tag that isn't
        currently held (never acquired, or already released) is a no-op,
        not an error, so callers can release defensively (e.g. in a
        `finally` block) without needing to track whether they actually
        acquired anything first. If dtype/device are omitted, matches
        every buffer registered under this tag (normally just one).
        """
        for key, buf in self._buffers.items():
            if key.tag == tag and (dtype is None or key.dtype == dtype) \
                    and (device is None or key.device == str(device)):
                buf.in_use = False

    def free(self, tag: str, dtype: torch.dtype | None = None, device=None) -> None:
        """Actually drop the reference to buffer(s) registered under
        `tag`, so the underlying device memory can be reclaimed by the
        allocator -- distinct from release() (see that method's
        docstring and this module's docstring for why the distinction
        matters). The next get_buffer() for this tag allocates fresh.
        """
        for key in [k for k in self._buffers
                    if k.tag == tag
                    and (dtype is None or k.dtype == dtype)
                    and (device is None or k.device == str(device))]:
            del self._buffers[key]

    def free_all(self) -> None:
        """Drop every buffer this manager is tracking, regardless of tag
        or in-use state. Intended for a caller's own free/discard
        lifecycle point (e.g. OptimizerHandle.free_states(), or a
        strategy's offload_extra() freeing VRAM before an offload round
        trip -- see strategies/chunked.py) -- a manager should never
        silently keep holding device memory past the point its owner
        believed it had been released.

        Also drops this manager's own references to any per-device
        MemPool it created (use_mempool=True only) -- since every buffer
        that pool backed is also being dropped here, this should let the
        pool's memory be reclaimed once nothing else references it.
        Genuinely uncertain about MemPool's exact teardown semantics
        beyond normal Python refcounting (no documented explicit
        "destroy" API was found) -- stated honestly rather than assumed;
        needs real-hardware confirmation, not just an assertion here that
        it works.
        """
        self._buffers.clear()
        self._mempools.clear()

    def stats(self) -> dict[str, Any]:
        """Per-tag byte usage plus a total, for diagnostics. This is the
        "visibility across everything this manager owns" property that
        motivated centralizing buffer management in the first place --
        with each strategy privately holding its own opaque
        `self._scratch` tensor, there was no single place to ask "how
        much device memory does this optimizer's scratch space actually
        use right now." Byte totals are summed across dtype/device
        variants of the same tag name.
        """
        per_tag: dict[str, int] = {}
        for key, buf in self._buffers.items():
            nbytes = buf.tensor.numel() * buf.tensor.element_size()
            per_tag[key.tag] = per_tag.get(key.tag, 0) + nbytes
        return {"per_tag_bytes": per_tag, "total_bytes": sum(per_tag.values())}
