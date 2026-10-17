"""How much of the card this process holds, as the kernel counts it.

The allocator knows only its own blocks; the driver's total includes the desktop and every other program. The kernel
keeps a statistic per open GPU handle and shows it under ``/proc/<pid>/fdinfo``, so the share of this process (blocks,
context, compiled kernels, primitive scratch) can be read exactly instead of inferred by subtraction.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_UNITS = {"": 1, "kib": 2**10, "mib": 2**20, "gib": 2**30}
_RESIDENT = re.compile(r"^drm-resident-(?:vram|local)\d+:\s*(\d+)\s*([KMG]iB)?\s*$", re.IGNORECASE | re.MULTILINE)
_CLIENT = re.compile(r"^drm-client-id:\s*(\d+)\s*$", re.MULTILINE)
_DEVICE = re.compile(r"^drm-pdev:\s*(\S+)\s*$", re.MULTILINE)


def parse_fdinfo(text: str) -> tuple[tuple[str, str], int] | None:
    """``((device, client id), bytes resident in device memory)`` of one handle, None when it is not a GPU handle
    that reports device memory."""
    client = _CLIENT.search(text)
    if client is None:
        return None
    found = _RESIDENT.findall(text)
    if not found:
        return None
    device = _DEVICE.search(text)
    return (device.group(1) if device else "", client.group(1)), sum(int(n) * _UNITS[unit.lower()] for n, unit in found)


class DriverMemory:
    """Reads the per-client statistics of this process. Handles that share one client are counted once."""

    def __init__(self, directory: str | Path = "/proc/self/fdinfo"):
        self.directory = Path(directory)

    def resident_mb(self) -> float | None:
        """Megabytes this process holds in device memory; None where the kernel does not say (other systems,
        drivers without the statistic)."""
        try:
            names = os.listdir(self.directory)
        except OSError:
            return None
        clients: dict[tuple[str, str], int] = {}
        for name in names:
            try:
                parsed = parse_fdinfo((self.directory / name).read_text(errors="replace"))
            except OSError:  # a handle closed while we looked
                continue
            if parsed is not None:
                clients[parsed[0]] = parsed[1]
        return round(sum(clients.values()) / 2**20, 1) if clients else None
