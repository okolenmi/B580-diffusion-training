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

    # ----- reading for a browser -----

    def count(self, set_name: str = DEFAULT_SET, *, query: str = "", missing: str = "") -> int:
        where, args = self._filter(set_name, query, missing)
        return self._db.execute(f"select count(*) from images i {where}", args).fetchone()[0]

    def page(self, set_name: str = DEFAULT_SET, *, query: str = "", missing: str = "", offset: int = 0, limit: int = 60) -> list[dict]:
        """Images of one set, newest ingest first, each with its captions and the sets it is in (besides ``all``).

        ``query`` matches any caption or the path; ``missing`` keeps only images without a caption in that channel.
        """
        where, args = self._filter(set_name, query, missing)
        rows = self._db.execute(
            f"select i.id, i.path, i.width, i.height, i.bytes from images i {where} order by i.added desc, i.id limit ? offset ?",
            (*args, limit, offset)).fetchall()
        caps: dict[str, dict[str, str]] = {r[0]: {} for r in rows}
        for image_id, channel, text in self._db.execute(
                f"select id, channel, text from captions where id in ({','.join('?' * len(rows))})", [r[0] for r in rows]) if rows else ():
            caps[image_id][channel] = text
        member_of: dict[str, list[str]] = {r[0]: [] for r in rows}
        for image_id, name in self._db.execute(
                f"select id, set_name from members where set_name != ? and id in ({','.join('?' * len(rows))}) order by set_name",
                [DEFAULT_SET, *member_of]) if rows else ():
            member_of[image_id].append(name)
        return [{"id": i, "path": p, "width": w, "height": h, "bytes": b, "captions": caps[i], "sets": member_of[i]}
                for i, p, w, h, b in rows]

    def path_of(self, image_id: str) -> str | None:
        row = self._db.execute("select path from images where id=?", (image_id,)).fetchone()
        return row[0] if row else None

    def _filter(self, set_name: str, query: str, missing: str = "") -> tuple[str, list]:
        where, args = "join members m on m.id=i.id where m.set_name=?", [set_name]
        if missing:
            if missing not in CHANNELS:
                raise ValueError(f"unknown caption channel {missing!r}; choose from {CHANNELS}")
            where += " and not exists(select 1 from captions c where c.id=i.id and c.channel=? and trim(c.text) != '')"
            args.append(missing)
        if query.strip():
            like = f"%{query.strip()}%"
            where += " and (i.path like ? or exists(select 1 from captions c where c.id=i.id and c.text like ?))"
            args += [like, like]
        return where, args

    def ids(self, set_name: str = DEFAULT_SET, *, query: str = "", missing: str = "") -> list[str]:
        where, args = self._filter(set_name, query, missing)
        return [r[0] for r in self._db.execute(f"select i.id from images i {where} order by i.added desc, i.id", args)]

    def stats(self) -> dict:
        images, = self._db.execute("select count(*) from images").fetchone()
        sets = {n: c for n, c in self._db.execute("select set_name, count(*) from members group by set_name")}
        captioned = {ch: n for ch, n in self._db.execute("select channel, count(*) from captions where trim(text) != '' group by channel")}
        total_bytes, = self._db.execute("select coalesce(sum(bytes), 0) from images").fetchone()
        shape = {k: n for k, n in self._db.execute(
            "select case when width*10 < height*9 then 'portrait' when width*9 > height*10 then 'landscape' else 'square' end, count(*) "
            "from images group by 1")}
        covers = [r[0] for r in self._db.execute("select id from images order by added desc, id limit 4")]
        return {"images": images, "bytes": total_bytes, "sets": {n: sets.get(n, 0) for n in self.sets()}, "captioned": captioned,
                "orientation": {k: shape.get(k, 0) for k in ("portrait", "square", "landscape")}, "covers": covers}

    # ----- editing -----

    def edit_captions(self, ids: Iterable[str], channel: str, mode: str, text: str = "", find: str = "") -> int:
        """Change one channel of many images: ``set`` | ``prepend`` | ``append`` | ``replace`` (``find`` -> ``text``) | ``clear``.

        ``prepend`` / ``append`` skip an image that already has the text there, so repeating a trigger word is harmless.
        """
        if mode not in ("set", "prepend", "append", "replace", "clear"):
            raise ValueError(f"unknown caption edit {mode!r}")
        if channel not in CHANNELS:
            raise ValueError(f"unknown caption channel {channel!r}; choose from {CHANNELS}")
        changed = 0
        for image_id in ids:
            row = self._db.execute("select text from captions where id=? and channel=?", (image_id, channel)).fetchone()
            old = row[0] if row else ""
            if mode == "set": new = text
            elif mode == "clear": new = ""
            elif mode == "replace": new = old.replace(find, text) if find else old
            elif mode == "prepend": new = old if old.startswith(text) else f"{text}{', ' if old and text else ''}{old}"
            else: new = old if old.endswith(text) else f"{old}{', ' if old and text else ''}{text}"
            if new != old:
                self.set_caption(image_id, new, channel, commit=False)
                changed += 1
        self._db.commit()
        return changed

    def remove(self, ids: Iterable[str], set_name: str | None = None) -> int:
        """Take images out of one set, or out of the library when ``set_name`` is None (their files stay on disk)."""
        ids = list(ids)
        if set_name in (None, DEFAULT_SET):
            self._db.executemany("delete from images where id=?", [(i,) for i in ids])
        else:
            self._db.executemany("delete from members where set_name=? and id=?", [(set_name, i) for i in ids])
        self._db.commit()
        return len(ids)

    def add_to_set(self, set_name: str, ids: Iterable[str]) -> int:
        self._db.execute("insert or ignore into sets(name) values(?)", (set_name,))
        self._db.executemany("insert or ignore into members values(?,?)", [(set_name, i) for i in ids])
        self._db.commit()
        return self._db.execute("select count(*) from members where set_name=?", (set_name,)).fetchone()[0]

    def delete_set(self, set_name: str) -> None:
        if set_name == DEFAULT_SET:
            raise ValueError(f"the {DEFAULT_SET!r} set holds every image and cannot be deleted")
        self._db.execute("delete from sets where name=?", (set_name,))
        self._db.commit()

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
    if channel not in CHANNELS:
        raise ValueError(f"unknown caption channel {channel!r}; choose from {CHANNELS}")
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
