"""Graphics cards as the operating system lists them, for the machine that has no torch yet.

The setup page asks which card the torch wheel is for before any wheel is installed, and torch is the only thing that can open a card.
This lists display adapters by name from the system instead (``lspci`` or sysfs on Linux, PowerShell on Windows) so the page can say what
it sees; whether a card is usable is checked by torch after the install. Nothing here may raise: no answer is an empty list.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_DISPLAY_CLASSES = ("VGA", "Display controller", "3D controller")
_VENDORS = {"0x8086": "Intel", "0x10de": "NVIDIA", "0x1002": "AMD"}

Runner = Callable[[list[str]], str]


@dataclass(frozen=True)
class SystemCard:
    name: str
    memory_mb: int | None = None


def _run(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    return done.stdout if done.returncode == 0 else ""


class SystemCards:
    """Lists display adapters without importing anything heavy. ``runner`` and ``drm_root`` are injectable for tests."""

    def __init__(self, platform: str | None = None, runner: Runner = _run, drm_root: Path = Path("/sys/class/drm")) -> None:
        self.platform = platform or sys.platform
        self.runner = runner
        self.drm_root = drm_root

    def list(self) -> tuple[SystemCard, ...]:
        try:
            if self.platform.startswith("win"):
                return self._windows()
            if self.platform.startswith("linux"):
                return self._lspci() or self._sysfs()
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
        return ()

    def _lspci(self) -> tuple[SystemCard, ...]:
        try:
            text = self.runner(["lspci", "-mm"])
        except (OSError, subprocess.SubprocessError):
            return ()  # no pciutils: the caller falls back to sysfs
        cards = []
        for line in text.splitlines():
            fields = shlex.split(line)
            if len(fields) >= 4 and fields[1].startswith(_DISPLAY_CLASSES):
                cards.append(SystemCard(f"{fields[2]} {fields[3]}".replace(" Corporation", "")))
        return tuple(cards)

    def _sysfs(self) -> tuple[SystemCard, ...]:
        cards = []
        for node in sorted(self.drm_root.glob("card[0-9]*/device")):
            vendor = (node / "vendor").read_text().strip() if (node / "vendor").is_file() else ""
            product = (node / "device").read_text().strip() if (node / "device").is_file() else ""
            if vendor:
                cards.append(SystemCard(f"{_VENDORS.get(vendor, 'GPU ' + vendor)} graphics (device {product or '?'})"))
        return tuple(cards)

    def _windows(self) -> tuple[SystemCard, ...]:
        script = "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
        text = self.runner(["powershell", "-NoProfile", "-Command", script]).strip()
        if not text:
            return ()
        rows = json.loads(text)
        rows = rows if isinstance(rows, list) else [rows]
        # AdapterRAM is a 32-bit field: a card with more than 4 GB reports a wrong number, so memory is left to torch.
        return tuple(SystemCard(str(row["Name"])) for row in rows if row.get("Name"))
