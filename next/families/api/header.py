"""Reading a safetensors header without torch, without the safetensors package and without data.

The format is documented and stable: eight bytes (little-endian u64) give the length of a JSON
header; the tensor data follows. Only the header is read, so detection never touches tensor
data, and the header length is capped so a hostile file cannot exhaust memory.
"""

from __future__ import annotations

import bisect
import json
import os
import struct
from collections import OrderedDict, Counter
from dataclasses import dataclass
from pathlib import Path

MAX_HEADER_BYTES = 100 * 1024 * 1024
PICKLE_SUFFIXES = {".ckpt", ".pt", ".pth", ".bin", ".pkl"}

DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1,
               "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}


class ModelFileError(Exception):
    """A file cannot be used as a model source. ``kind`` is machine-readable."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


class HeaderIndex:
    """Key names, shapes and dtypes of one file. Never holds tensor data."""

    def __init__(self, entries: dict[str, TensorInfo], metadata: dict[str, str], file_size: int,
                 header_bytes: int, path: str = ""):
        self.entries = entries
        self.metadata = metadata
        self.file_size = file_size
        self.header_bytes = header_bytes
        self.path = path
        self._sorted = sorted(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, key: str) -> bool:
        return key in self.entries

    def shape(self, key: str) -> tuple[int, ...] | None:
        info = self.entries.get(key)
        return info.shape if info else None

    def dtype(self, key: str) -> str | None:
        info = self.entries.get(key)
        return info.dtype if info else None

    def keys_with(self, prefix: str) -> list[str]:
        start = bisect.bisect_left(self._sorted, prefix)
        out = []
        for key in self._sorted[start:]:
            if not key.startswith(prefix):
                break
            out.append(key)
        return out

    def has_prefix(self, prefix: str) -> bool:
        start = bisect.bisect_left(self._sorted, prefix)
        return start < len(self._sorted) and self._sorted[start].startswith(prefix)

    def count(self, prefix: str) -> int:
        return len(self.keys_with(prefix))

    def nbytes(self, prefix: str = "") -> int:
        return sum(self.entries[k].nbytes for k in self.keys_with(prefix))

    def dominant_dtype(self, prefix: str = "") -> str | None:
        counts: Counter = Counter()
        for key in self.keys_with(prefix):
            info = self.entries[key]
            counts[info.dtype] += max(1, info.nbytes)
        return counts.most_common(1)[0][0] if counts else None

    def top_prefixes(self, depth: int = 2, limit: int = 12) -> list[tuple[str, int]]:
        """('a.b.', key count) for the largest groups at ``depth`` dotted segments: for error reports."""
        counts: Counter = Counter()
        for key in self.entries:
            counts[".".join(key.split(".")[:depth]) + "."] += 1
        return counts.most_common(limit)


def read_header(path: str | os.PathLike, max_header: int = MAX_HEADER_BYTES) -> HeaderIndex:
    p = Path(path)
    if p.suffix.lower() in PICKLE_SUFFIXES:
        raise ModelFileError("pickle_refused",
                             f"{p.name}: pickle files can execute code when loaded; convert it to .safetensors")
    if p.suffix.lower() != ".safetensors":
        raise ModelFileError("not_safetensors", f"{p.name}: only .safetensors files are supported")
    try:
        size = p.stat().st_size
        with open(p, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise ModelFileError("bad_header", f"{p.name}: too short to be a safetensors file")
            (length,) = struct.unpack("<Q", prefix)
            if length == 0 or length > max_header:
                raise ModelFileError("header_too_large" if length > max_header else "bad_header",
                                     f"{p.name}: header length {length} is not valid (limit {max_header})")
            if 8 + length > size:
                raise ModelFileError("truncated", f"{p.name}: header runs past the end of the file")
            raw = handle.read(length)
    except FileNotFoundError:
        raise ModelFileError("not_found", f"{p.name}: file not found") from None
    except OSError as exc:
        raise ModelFileError("unreadable", f"{p.name}: {exc}") from None
    try:
        header = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelFileError("bad_header", f"{p.name}: header is not valid JSON ({exc})") from None
    if not isinstance(header, dict):
        raise ModelFileError("bad_header", f"{p.name}: header is not an object")
    metadata = header.pop("__metadata__", {})
    metadata = {str(k): str(v) for k, v in metadata.items()} if isinstance(metadata, dict) else {}
    entries: dict[str, TensorInfo] = {}
    data_start = 8 + length
    furthest = 0
    for name, info in header.items():
        try:
            dtype = info["dtype"]
            shape = tuple(int(n) for n in info["shape"])
            begin, end = (int(n) for n in info["data_offsets"])
        except (KeyError, TypeError, ValueError):
            raise ModelFileError("bad_header", f"{p.name}: malformed entry for tensor {name!r}") from None
        if begin < 0 or end < begin:
            raise ModelFileError("bad_header", f"{p.name}: bad data offsets for tensor {name!r}")
        furthest = max(furthest, end)
        entries[name] = TensorInfo(str(dtype), shape, end - begin)
    if data_start + furthest > size:
        raise ModelFileError("truncated", f"{p.name}: file is shorter than its header says (truncated download?)")
    return HeaderIndex(entries, metadata, size, length, str(p))


class HeaderCache:
    """LRU of parsed headers keyed by (real path, size, mtime_ns); a changed file is re-read."""

    def __init__(self, capacity: int = 64):
        self._capacity = capacity
        self._items: OrderedDict[tuple, HeaderIndex] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, path: str | os.PathLike) -> HeaderIndex:
        real = os.path.realpath(path)
        try:
            stat = os.stat(real)
            key = (real, stat.st_size, stat.st_mtime_ns)
        except OSError:
            return read_header(real)  # raises the right ModelFileError
        found = self._items.get(key)
        if found is not None:
            self._items.move_to_end(key)
            self.hits += 1
            return found
        self.misses += 1
        index = read_header(real)
        self._items[key] = index
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)
        return index
