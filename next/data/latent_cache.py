"""Latents, cached per VAE.

A cache directory is named by the VAE's identity (``ids.vae_identity``) and records the latent spec it
was made with. Opening it with a different identity or spec is refused, so a dataset encoded for one
latent space can never reach a model that expects another. Entries are keyed by (sample, size
policy); an image is encoded once per policy, and only missing images are encoded on the next build.

Latents are stored as float16 in append-only safetensors shards and read back as float32.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .ids import image_of, sample_id, spec_fields
from .images import SizePolicy, load_crops
from .library import ImageSet
from .prefetch import Prefetch

SHARD_SAMPLES = 256
INDEX = "index.sqlite"

_SCHEMA = """
create table if not exists meta(key text primary key, value text not null);
create table if not exists done(image text not null, policy text not null, crops integer not null,
    primary key(image, policy));
create table if not exists entries(sample text not null, policy text not null, shard text not null,
    h integer not null, w integer not null, primary key(sample, policy));
"""


class CacheMismatch(Exception):
    """The cache belongs to a different VAE or latent space than the one asked for."""


class Encoder(Protocol):
    def encode(self, images: torch.Tensor) -> torch.Tensor: ...


@dataclass(frozen=True)
class CacheInfo:
    vae_id: str
    label: str
    spec: dict
    samples: int
    bytes: int


@dataclass(frozen=True)
class BuildStats:
    encoded_images: int
    encoded_samples: int
    skipped: list[tuple[str, str]]
    stopped: bool = False


class LatentCache:
    def __init__(self, root: str | Path, vae_id: str, spec: Any, label: str = ""):
        self.vae_id = vae_id
        self.spec = spec_fields(spec)
        self.dir = Path(root) / vae_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.dir / INDEX, check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self._open: dict[str, Any] = {}
        stored = dict(self._db.execute("select key, value from meta"))
        if stored:
            self._check(stored, label)
        else:
            self._db.executemany("insert into meta values(?,?)", [
                ("vae_id", vae_id), ("spec", json.dumps(self.spec, sort_keys=True)), ("label", label)])
            self._db.commit()

    def _check(self, stored: dict, label: str) -> None:
        if stored["vae_id"] != self.vae_id or json.loads(stored["spec"]) != self.spec:
            raise CacheMismatch(
                f"this cache holds latents of {stored.get('label') or stored['vae_id']} "
                f"({stored['spec']}), but the graph's VAE is {label or self.vae_id} ({json.dumps(self.spec)})")

    def close(self) -> None:
        with self._lock:
            for handle in self._open.values():
                handle.__exit__(None, None, None)
            self._open.clear()
        self._db.close()

    def missing(self, images: ImageSet, policy: SizePolicy) -> list[str]:
        done = {r[0] for r in self._db.execute("select image from done where policy=?", (policy.key,))}
        return [i for i in images.ids if i not in done]

    def build(self, images: ImageSet, policy: SizePolicy, encoder: Encoder, *, batch_size: int = 4,
              progress: Callable[[int, int], None] | None = None,
              should_stop: Callable[[], bool] | None = None,
              recover: Callable[[], None] | None = None) -> BuildStats:
        """Encode every image that has no entry for ``policy``; safe to interrupt and run again.

        When the device runs out of memory the batch is halved and retried (``recover`` runs first, to give back
        cached memory); only a single image that still does not fit is an error.
        """
        todo = self.missing(images, policy)
        return _Encoding(self, images, policy, encoder, batch_size, recover).run(todo, progress, should_stop)

    def _write_shard(self, latents: dict[str, torch.Tensor], policy: SizePolicy, crops_of: dict[str, int]) -> None:
        name = f"shard_{len(list(self.dir.glob('shard_*.safetensors'))):05d}.safetensors"
        tmp = self.dir / (name + ".tmp")
        save_file({k: v.contiguous() for k, v in latents.items()}, str(tmp))
        os.replace(tmp, self.dir / name)
        rows = [(s, policy.key, name, int(v.shape[-2]), int(v.shape[-1])) for s, v in latents.items()]
        by_image: dict[str, int] = {}
        for s in latents:
            by_image[image_of(s)] = by_image.get(image_of(s), 0) + 1
        with self._lock:
            self._db.executemany("insert or replace into entries values(?,?,?,?,?)", rows)
            # An image counts as done only when all of its crops landed in the index.
            for image, count in by_image.items():
                stored = self._db.execute("select count(*) from entries where policy=? and (sample=? or sample like ?)",
                                          (policy.key, image, image + "#%")).fetchone()[0]
                if stored >= crops_of.get(image, count):
                    self._db.execute("insert or replace into done values(?,?,?)", (image, policy.key, stored))
            self._db.commit()

    def view(self, images: ImageSet, policy: SizePolicy, *, skip_missing: bool = False) -> LatentSet:
        """The samples of ``images`` for ``policy``. Missing latents are an error here, not at step 4000,
        unless the caller already knows some images could not be encoded and passes ``skip_missing``."""
        todo = [] if skip_missing else self.missing(images, policy)
        if todo:
            raise LookupError(f"{len(todo)} image(s) have no latents for policy {policy.key} in cache {self.vae_id} "
                              f"(first: {todo[0]}); build the cache first")
        rows = self._db.execute("select sample, shard, h, w from entries where policy=?", (policy.key,)).fetchall()
        wanted = set(images.ids)
        entries = {r[0]: (r[1], (r[2], r[3])) for r in rows if image_of(r[0]) in wanted}
        return LatentSet(self, policy, tuple(sorted(entries)), {s: e[1] for s, e in entries.items()},
                         {s: e[0] for s, e in entries.items()})

    def read(self, shard: str, sample: str) -> torch.Tensor:
        with self._lock:
            handle = self._open.get(shard)
            if handle is None:
                handle = safe_open(str(self.dir / shard), framework="pt")
                handle.__enter__()
                self._open[shard] = handle
            return handle.get_tensor(sample).float()

    def info(self, label: str = "") -> CacheInfo:
        count = self._db.execute("select count(*) from entries").fetchone()[0]
        size = sum(p.stat().st_size for p in self.dir.glob("shard_*.safetensors"))
        stored = dict(self._db.execute("select key, value from meta"))
        return CacheInfo(self.vae_id, stored.get("label", label), json.loads(stored["spec"]), count, size)


@dataclass(frozen=True)
class LatentSet:
    """Handle to the latents of one image set under one policy. Immutable; reads go to the cache."""

    cache: LatentCache
    policy: SizePolicy
    samples: tuple[str, ...]
    shapes: dict[str, tuple[int, int]]  # latent (h, w)
    shards: dict[str, str]

    @property
    def spec(self) -> dict:
        return self.cache.spec

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, sample: str) -> torch.Tensor:
        return self.cache.read(self.shards[sample], sample)


def list_caches(root: str | Path) -> list[CacheInfo]:
    """What a cache folder holds, for the Datasets page: which VAEs have latents, and how big they are."""
    found = []
    for child in sorted(Path(root).glob("*/" + INDEX)) if Path(root).is_dir() else []:
        db = sqlite3.connect(child)
        try:
            meta = dict(db.execute("select key, value from meta"))
            count = db.execute("select count(*) from entries").fetchone()[0]
        finally:
            db.close()
        size = sum(p.stat().st_size for p in child.parent.glob("shard_*.safetensors"))
        found.append(CacheInfo(meta["vae_id"], meta.get("label", ""), json.loads(meta["spec"]), count, size))
    return found


class _Encoding:
    """One ``build``: images are decoded ahead on a thread, grouped by shape into batches for the encoder, and written
    in shards. The batch size only goes down: after an out-of-memory error it stays at the size that fit."""

    def __init__(self, cache: LatentCache, images: ImageSet, policy: SizePolicy, encoder: Encoder, batch_size: int,
                 recover: Callable[[], None] | None) -> None:
        self.cache, self.images, self.policy, self.encoder, self.recover = cache, images, policy, encoder, recover
        self.limit = batch_size
        self.pending: list[tuple[str, torch.Tensor]] = []  # (sample, uint8 CHW)
        self.encoded: dict[str, torch.Tensor] = {}  # waiting for the next shard
        self.crops_of: dict[str, int] = {}
        self.skipped: list[tuple[str, str]] = []
        self.images_done = self.samples_done = 0

    def run(self, todo: list[str], progress: Callable[[int, int], None] | None,
            should_stop: Callable[[], bool] | None) -> BuildStats:
        loader = Prefetch(lambda: map(self._load, todo), depth=8)  # decoding runs ahead while the device encodes
        try:
            for n, (image_id, crops, error) in enumerate(loader, 1):
                if should_stop and should_stop():
                    self._flush(True)
                    return self._stats(stopped=True)
                if error is not None:
                    self.skipped.append((self.images.paths[image_id], f"{type(error).__name__}: {error}"))
                    continue
                self.crops_of[image_id] = len(crops)
                for k, crop in enumerate(crops):
                    self.pending.append((sample_id(image_id, k), torch.from_numpy(crop.copy()).permute(2, 0, 1).contiguous()))
                self.images_done += 1
                self._flush(False)
                if progress:
                    progress(n, len(todo))
        finally:
            loader.close()
        self._flush(True)
        return self._stats()

    def _stats(self, stopped: bool = False) -> BuildStats:
        return BuildStats(self.images_done, self.samples_done, self.skipped, stopped=stopped)

    def _load(self, image_id: str):
        try:
            return image_id, load_crops(self.images.paths[image_id], self.policy), None
        except Exception as exc:  # noqa: BLE001 - reported as a skipped image
            return image_id, None, exc

    def _encode(self, chunk: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
        """Encode a chunk; when the device runs out of memory, give cached memory back and encode it in halves."""
        try:
            return self.encoder.encode(torch.stack([c[1] for c in chunk])).to("cpu", torch.float16)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.OutOfMemoryError) and "out of memory" not in str(exc).lower():
                raise
            if self.recover is not None:
                self.recover()
            if len(chunk) == 1:
                raise RuntimeError(f"one {chunk[0][1].shape[-1]}x{chunk[0][1].shape[-2]} image does not fit on the device "
                                   f"for encoding; lower the size, or encode on the CPU") from exc
            half = len(chunk) // 2
            self.limit = max(1, half)
            return torch.cat([self._encode(chunk[:half]), self._encode(chunk[half:])])

    def _flush(self, force: bool) -> None:
        """Encode every full batch of one shape (all of them when ``force``) and write a shard when enough is ready."""
        by_shape: dict[tuple[int, int], list[tuple[str, torch.Tensor]]] = {}
        for item in self.pending:
            by_shape.setdefault(tuple(item[1].shape[-2:]), []).append(item)
        rest = []
        for group in by_shape.values():
            while len(group) >= self.limit or (force and group):
                chunk, group = group[:self.limit], group[self.limit:]
                latents = self._encode(chunk)
                self.encoded.update({c[0]: latents[k] for k, c in enumerate(chunk)})
            rest += group
        self.pending = rest
        if len(self.encoded) >= SHARD_SAMPLES or (force and self.encoded):
            self.cache._write_shard(self.encoded, self.policy, self.crops_of)
            self.samples_done += len(self.encoded)
            self.encoded.clear()
