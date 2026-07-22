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

What this does NOT do (real, separate, honestly-out-of-scope follow-up):
no torch.xpu.MemPool integration yet -- get_buffer()'s single
torch.empty() call is exactly the seam that would wrap, later, without
any caller of this class needing to change. No cross-process or
cross-device-transfer support. No automatic eviction under memory
pressure -- a manager holds whatever it's been asked to hold until told
to release/free it, on purpose, so behavior stays predictable rather than
depending on runtime memory conditions.
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
    See module docstring for the full reasoning.
    """

    def __init__(self) -> None:
        self._buffers: dict[_BufferKey, _Buffer] = {}

    def get_buffer(self, tag: str, numel: int, dtype: torch.dtype, device) -> torch.Tensor:
        """Return a 1-D tensor of at least `numel` elements registered
        under `tag` (for this dtype/device). Reuses the existing buffer
        if one is already large enough; grows (reallocates) if not;
        never shrinks, so a tag's buffer only gets reallocated as often
        as its largest-ever request changes.

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
            buf = _Buffer(tensor=torch.empty(numel, dtype=dtype, device=device))
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
        """
        self._buffers.clear()

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
