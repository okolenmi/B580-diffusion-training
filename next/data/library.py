"""The image library: images and their captions, independent of any model.

One SQLite file per library. Images are identified by content hash (``ids.hash_file``), so moving a
folder or re-ingesting keeps captions and latent caches attached. Captions live in *channels*
(``text`` by default, plus ``tags`` and ``desc``), so a captioner can write one channel without
overwriting another. ``ImageSet`` is an immutable snapshot that the rest of the pipeline reads.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Iterable, Mapping

from .ids import hash_file

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CHANNELS = ("text", "tags", "desc")
SIDECAR = {"text": ".txt", "tags": ".tags.txt", "desc": ".desc.txt"}
DEFAULT_SET = "all"
TAG_SPLIT = re.compile(r"\s*,\s*")

_SCHEMA = """
create table if not exists images(
    id text primary key, path text not null, width integer not null, height integer not null,
    bytes integer not null, added real not null);
create table if not exists captions(
    id text not null references images(id) on delete cascade, channel text not null, text text not null,
    primary key(id, channel));
create table if not exists sets(name text primary key);
create table if not exists members(
    set_name text not null references sets(name) on delete cascade,
    id text not null references images(id) on delete cascade, primary key(set_name, id));
"""


def split_tags(caption: str) -> list[str]:
    return [t for t in TAG_SPLIT.split(caption.strip()) if t]


@dataclass(frozen=True)
class ImageSet:
    """Metadata only: no pixels, no latents. ``captions[channel][id]`` may be missing or empty."""

    ids: tuple[str, ...]
    paths: Mapping[str, str]
    sizes: Mapping[str, tuple[int, int]]  # (width, height)
    captions: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    def caption(self, image_id: str, channel: str = "text") -> str:
        return self.captions.get(channel, {}).get(image_id, "")

    def subset(self, ids: Iterable[str]) -> ImageSet:
        keep = tuple(i for i in ids if i in self.paths)
        return ImageSet(keep, {i: self.paths[i] for i in keep}, {i: self.sizes[i] for i in keep},
                        {c: {i: t for i, t in m.items() if i in set(keep)} for c, m in self.captions.items()})


@dataclass(frozen=True)
class IngestResult:
    added: int
    known: int
    skipped: list[tuple[str, str]]  # (path, reason)


class ImageLibrary:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("pragma foreign_keys = on")
        self._db.executescript(_SCHEMA)
        self._db.execute("insert or ignore into sets(name) values(?)", (DEFAULT_SET,))
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def ingest(self, folder: str | Path, *, set_name: str = DEFAULT_SET, recursive: bool = True,
               progress: Callable[[int, int], None] | None = None) -> IngestResult:
        """Register every image under ``folder`` and read sidecar captions (``a.txt``, ``a.tags.txt``).

        Captions already in the library are replaced only by a sidecar that exists; an image without
        sidecars keeps whatever it had.
        """
        from PIL import Image

        root = Path(folder)
        files = sorted(p for p in (root.rglob("*") if recursive else root.glob("*"))
                       if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
        self._db.execute("insert or ignore into sets(name) values(?)", (set_name,))
        added = known = 0
        skipped: list[tuple[str, str]] = []
        for n, file in enumerate(files, 1):
            try:
                image_id = hash_file(file)
                exists = self._db.execute("select 1 from images where id=?", (image_id,)).fetchone()
                if exists:
                    known += 1
                    self._db.execute("update images set path=? where id=?", (str(file), image_id))
                else:
                    with Image.open(file) as image:
                        width, height = image.size
                    self._db.execute("insert into images values(?,?,?,?,?,?)",
                                     (image_id, str(file), width, height, file.stat().st_size, time.time()))
                    added += 1
                for channel, suffix in SIDECAR.items():
                    sidecar = file.with_name(file.stem + suffix)
                    if sidecar.is_file():
                        self.set_caption(image_id, sidecar.read_text(encoding="utf-8").strip(), channel, commit=False)
                self._db.execute("insert or ignore into members values(?,?)", (set_name, image_id))
            except Exception as exc:  # one unreadable image must not stop the folder
                skipped.append((str(file), f"{type(exc).__name__}: {exc}"))
            if progress:
                progress(n, len(files))
        self._db.commit()
        return IngestResult(added, known, skipped)

    def set_caption(self, image_id: str, text: str, channel: str = "text", *, commit: bool = True) -> None:
        if channel not in CHANNELS:
            raise ValueError(f"unknown caption channel {channel!r}; choose from {CHANNELS}")
        self._db.execute("insert into captions values(?,?,?) on conflict(id, channel) do update set text=excluded.text",
                         (image_id, channel, text))
        if commit:
            self._db.commit()

    def commit(self) -> None:
        self._db.commit()

    def sets(self) -> list[str]:
        return [r[0] for r in self._db.execute("select name from sets order by name")]

    def snapshot(self, set_name: str = DEFAULT_SET, *, min_side: int = 0, caption_required: str | None = None) -> ImageSet:
        """An immutable ``ImageSet`` of one set, optionally filtered by size or by a required caption channel."""
        rows = self._db.execute(
            "select i.id, i.path, i.width, i.height from images i join members m on m.id=i.id "
            "where m.set_name=? and min(i.width, i.height) >= ? order by i.id", (set_name, min_side)).fetchall()
        caps: dict[str, dict[str, str]] = {c: {} for c in CHANNELS}
        for image_id, channel, text in self._db.execute(
                "select c.id, c.channel, c.text from captions c join members m on m.id=c.id where m.set_name=?",
                (set_name,)):
            caps[channel][image_id] = text
        if caption_required:
            rows = [r for r in rows if caps.get(caption_required, {}).get(r[0], "").strip()]
        ids = tuple(r[0] for r in rows)
        return ImageSet(ids, {r[0]: r[1] for r in rows}, {r[0]: (r[2], r[3]) for r in rows},
                        {c: {i: t for i, t in m.items() if i in set(ids)} for c, m in caps.items() if m})


def report(images: ImageSet, channel: str = "text", top_tags: int = 30) -> dict:
    """Problems worth seeing before a long run: size spread, empty captions, repeated captions, tag counts."""
    sizes = Counter(images.sizes[i] for i in images.ids)
    captions = {i: images.caption(i, channel).strip() for i in images.ids}
    same: dict[str, list[str]] = {}
    for image_id, text in captions.items():
        if text:
            same.setdefault(text, []).append(image_id)
    tags = Counter(t for text in captions.values() for t in split_tags(text))
    short_sides = sorted(min(images.sizes[i]) for i in images.ids)
    return {
        "images": len(images),
        "sizes": [{"width": w, "height": h, "count": n} for (w, h), n in sizes.most_common(20)],
        "distinct_sizes": len(sizes),
        "smallest_side": short_sides[0] if short_sides else 0,
        "empty_captions": [i for i, t in captions.items() if not t],
        "repeated_captions": [{"text": t[:80], "count": len(v)} for t, v in same.items() if len(v) > 1][:20],
        "distinct_captions": len(same),
        "tags": tags.most_common(top_tags),
        "distinct_tags": len(tags),
    }
