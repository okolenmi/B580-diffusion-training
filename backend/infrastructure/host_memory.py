"""Host (CPU) memory from ``/proc`` -- the machine's, and one process's.

Two different questions, kept apart: how much RAM the machine can still give (``ProcHostMemory``, the server's admission check)
and how much one process holds (``process_held_mb``, the child's own ceiling guard). A platform without ``/proc`` answers
None to both; callers treat that as "cannot tell", never as zero.
"""

from __future__ import annotations

from pathlib import Path

from ..application.ports.host_memory import HostMemory, HostRam

_KB_PER_MB = 1024.0


def _kb_fields(text: str) -> dict[str, float]:
    """``Key:   123 kB`` lines as ``{"Key": 123.0}`` (other lines are skipped)."""
    fields: dict[str, float] = {}
    for line in text.splitlines():
        key, sep, rest = line.partition(":")
        parts = rest.split()
        if sep and len(parts) >= 1:
            try:
                fields[key.strip()] = float(parts[0])
            except ValueError:
                continue
    return fields


def parse_meminfo(text: str) -> HostRam | None:
    fields = _kb_fields(text)
    total, available = fields.get("MemTotal"), fields.get("MemAvailable")
    if not total or available is None:
        return None
    return HostRam(total_mb=total / _KB_PER_MB, available_mb=available / _KB_PER_MB)


def parse_held_mb(status_text: str) -> float | None:
    """What a process holds that the kernel cannot simply drop: private and shared anonymous memory.

    File-backed pages (a memory-mapped checkpoint) are left out because they are reclaimable and would count a model file
    twice; a kernel that does not split the resident set falls back to the whole of it.
    """
    fields = _kb_fields(status_text)
    if "RssAnon" in fields:
        return (fields["RssAnon"] + fields.get("RssShmem", 0.0)) / _KB_PER_MB
    if "VmRSS" in fields:
        return fields["VmRSS"] / _KB_PER_MB
    return None


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="ascii", errors="replace")
    except OSError:
        return None


class ProcHostMemory(HostMemory):
    def snapshot(self) -> HostRam | None:
        text = _read("/proc/meminfo")
        return None if text is None else parse_meminfo(text)


def process_held_mb(pid: int | str = "self") -> float | None:
    text = _read(f"/proc/{pid}/status")
    return None if text is None else parse_held_mb(text)
