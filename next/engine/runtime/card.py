"""Reading the card: the driver's numbers split into who holds what, and the ledger that accounts for every megabyte."""

from __future__ import annotations

from collections.abc import Mapping

from .residents import Claim

MB = 2**20


def outside(stats: Mapping[str, float]) -> tuple[float | None, float]:
    """``(this process's memory outside the allocator, everything else on the card)``; the first is None when the
    kernel does not report this process's share, and then the second is the whole outside."""
    outside_mb = max(stats["device_used"] - stats.get("reserved", 0.0), 0.0)
    if "process" not in stats:
        return None, outside_mb
    own = max(stats["process"] - stats.get("reserved", 0.0), 0.0)
    return own, max(outside_mb - own, 0.0)


def outside_or_unknown(stats: Mapping[str, float]) -> tuple[float | None, float]:
    """``outside`` for stats that may come without driver numbers (then nothing is outside as far as anyone can tell)."""
    return outside(stats) if "device_used" in stats else (None, 0.0)


def returnable_mb(stats: Mapping[str, float]) -> float:
    """What emptying the cache can really give back: free memory in whole segments, not pieces between live blocks."""
    free = max(stats.get("reserved", 0.0) - stats.get("allocated", 0.0), 0.0)
    return max(free - stats.get("split_free", 0.0), 0.0)


def need_mb(stats: Mapping[str, float]) -> float | None:
    """What the step needs that cannot be given back: its peak live tensors plus what the driver holds outside the
    allocator. Free blocks the allocator keeps are left out because it releases them when an allocation would fail."""
    if "device_used" not in stats:
        return None
    if "peak_allocated" not in stats or "reserved" not in stats:
        return stats["device_used"]
    return stats["peak_allocated"] + max(stats["device_used"] - stats["reserved"], 0.0)


def ledger_rows(claims: Mapping[str, Claim], stats: Mapping[str, float]) -> list[dict]:
    """Every megabyte the card holds, in exactly one row: claims, live tensors no claim covers, the allocator's
    free memory, what this process holds outside the allocator, and other programs. The monitor shows these rows."""
    rows = [{"name": name, "mb": round(c.resident.footprint_bytes() / MB, 1), "purpose": c.purpose} for name, c in claims.items()]
    if "allocated" not in stats:
        return rows
    claimed = sum(r["mb"] for r in rows)
    rows.append({"name": "unclaimed live", "mb": round(max(stats["allocated"] - claimed, 0.0), 1),
                 "purpose": "live tensors no claim covers (inputs and outputs of the step in flight, cached prompts, leftovers)"})
    free = max(stats.get("reserved", 0.0) - stats["allocated"], 0.0)
    if "split_free" in stats:
        stuck = min(stats["split_free"], free)
        rows.append({"name": "allocator free, returnable", "mb": round(free - stuck, 1),
                     "purpose": "whole free segments; handed back when the card needs them"})
        rows.append({"name": "allocator free, stuck", "mb": round(stuck, 1),
                     "purpose": "free pieces between live blocks; they stay while a neighbour lives (fragmentation)"})
    else:
        rows.append({"name": "allocator free blocks", "mb": round(free, 1), "purpose": "memory the allocator keeps for reuse"})
    if "device_used" in stats:
        own, other = outside(stats)
        if own is None:
            rows.append({"name": "outside allocator", "mb": round(other, 1),
                         "purpose": "this process (context, kernels, primitives) and other programs; the kernel does not say how it splits"})
        else:
            rows.append({"name": "outside allocator (ours)", "mb": round(own, 1),
                         "purpose": "this process's context, compiled kernels, primitive scratch and driver pools"})
            rows.append({"name": "other programs", "mb": round(other, 1), "purpose": "the desktop and everything else on the card"})
    return rows
