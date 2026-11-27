"""ExecutionLogs on the supervisor's scratch folder: ``execution_<id>.log``."""

from __future__ import annotations

from pathlib import Path


class FileExecutionLogs:
    def __init__(self, scratch_dir: Path) -> None:
        self._scratch = scratch_dir

    def tail(self, execution_id: int, max_bytes: int) -> str | None:
        path = self._scratch / f"execution_{int(execution_id)}.log"
        try:
            size = path.stat().st_size
            with path.open("rb") as handle:
                if size > max_bytes:
                    handle.seek(size - max_bytes)
                raw = handle.read()
        except OSError:
            return None
        text = raw.decode("utf-8", errors="replace")
        return text if size <= max_bytes else "[earlier output left out]\n" + text.split("\n", 1)[-1]
