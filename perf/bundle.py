"""A zip of the text a run left behind, small enough to send: logs cut in the middle, big json left out (its markdown twin stays), no traces or tensors."""

from __future__ import annotations

import zipfile
from pathlib import Path

LOG_CAP = 200_000
JSON_CAP = 1_000_000
TEXT_SUFFIXES = (".md", ".json", ".log")


def pack(folder: Path, name: str = "bundle.zip") -> Path:
    path = folder / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(folder.rglob("*")):
            if not file.is_file() or file == path or file.suffix not in TEXT_SUFFIXES or file.name == "trace.json":
                continue
            data = file.read_bytes()
            if file.suffix == ".json" and len(data) > JSON_CAP:
                continue
            if file.suffix == ".log" and len(data) > LOG_CAP:
                data = data[:LOG_CAP // 4] + b"\n... (middle cut) ...\n" + data[-LOG_CAP * 3 // 4:]
            archive.writestr(str(file.relative_to(folder)), data)
    return path
