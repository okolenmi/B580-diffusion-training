"""Captioning a managed dataset: fill in the captions it lacks.

A dataset ingested from images with no ``.txt`` sidecars stores an empty
caption per image, so training presents one constant conditioning -- the
captionless-conditioning problem (archive/bad-LoRA-investigation/MEASURED-
captionless-conditioning-break.md). This writes real captions into the
dataset's ``trajectories.prompt`` column, which is what the loader reads.

Pieces, all swappable:

* ``Captioner`` -- the one interface a caption model implements:
  ``caption(PIL.Image) -> str``. Nothing else in the pipeline knows which model
  produced a caption.
* ``CAPTIONER_BACKENDS`` / ``register_captioner`` -- name -> factory. A new
  backend (another local model, an API) is one registered factory; the node's
  ``backend`` port picks it by name. Shipped: ``"transformers"``, any Hugging
  Face image-to-text model (BLIP by default), run locally.
* ``caption_dataset`` -- the loop: find each image (the original, via the
  source's recorded image directory, else the dataset's own VAE preview),
  caption it, write the DB row. Resumable by construction (only empty captions
  are touched unless ``overwrite``), cancellable, and it commits as it goes so
  a stop keeps what was done.

Nothing here has run against a real caption model in the environment it was
written in; the loop is tested with a stub captioner, the transformers backend
is not (no model weights reachable there).

No torch import at module level; manager/ does not import nodes/.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .db import _connect, ensure_v2


class Captioner(ABC):

    @abstractmethod
    def caption(self, image) -> str:
        """A caption for one PIL image (RGB). May be empty if the model
        produced nothing; the caller decides what to do with that."""

    def close(self) -> None:
        """Release the model. Default: nothing to release."""


CAPTIONER_BACKENDS: dict[str, Callable[..., Captioner]] = {}


def register_captioner(name: str, factory: Callable[..., Captioner]) -> None:
    """Add (or replace) a backend. ``factory(model_id=..., device=...,
    max_new_tokens=...)`` must return a Captioner; it may ignore arguments
    that mean nothing to it."""
    CAPTIONER_BACKENDS[name] = factory


class TransformersCaptioner(Captioner):
    """Any Hugging Face ``image-to-text`` model, run locally (BLIP, GIT, ...).

    Imports transformers when constructed, not at import, so a graph that never
    uses captioning does not need it installed. ``device`` "auto" = the
    accelerator torch reports (xpu, then cuda), else CPU.
    """

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base",
                 device: str = "auto", max_new_tokens: int = 40, **_ignored):
        try:
            import torch
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'transformers' captioner needs the transformers package "
                "(pip install transformers). Original error: " + str(exc)) from exc
        if device == "auto":
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                device = "xpu"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._max_new_tokens = int(max_new_tokens)
        self._pipe = pipeline("image-to-text", model=model_id, device=device)

    def caption(self, image) -> str:  # pragma: no cover - needs real weights
        out = self._pipe(image, max_new_tokens=self._max_new_tokens)
        return (out[0].get("generated_text") or "").strip() if out else ""

    def close(self) -> None:  # pragma: no cover
        self._pipe = None


register_captioner("transformers", TransformersCaptioner)


def make_captioner(backend: str, **options) -> Captioner:
    if backend not in CAPTIONER_BACKENDS:
        raise ValueError(
            f"unknown captioner backend {backend!r}; registered: "
            f"{sorted(CAPTIONER_BACKENDS)}")
    return CAPTIONER_BACKENDS[backend](**options)


def clean_caption(text: str, prefix: str = "", suffix: str = "",
                  lowercase: bool = False) -> str:
    """Tidy a model's output into a training caption: collapse whitespace,
    drop a trailing divider, optionally lowercase, add trigger words around
    it. An empty model output stays empty (no bare prefix/suffix is written
    for an image the model said nothing about)."""
    text = " ".join((text or "").split()).rstrip(" .,")
    if not text:
        return ""
    if lowercase:
        text = text.lower()
    parts = [p.strip().rstrip(" ,") for p in (prefix, text, suffix)
             if p and p.strip()]
    return ", ".join(parts)


@dataclass
class CaptionReport:
    total: int = 0
    captioned: int = 0
    skipped_has_caption: int = 0
    missing_image: int = 0
    failed: int = 0
    empty_output: int = 0
    cancelled: bool = False
    used_preview: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{self.captioned} captioned of {self.total}"]
        for label, n in (("kept existing", self.skipped_has_caption),
                         ("from VAE preview", self.used_preview),
                         ("image not found", self.missing_image),
                         ("model said nothing", self.empty_output),
                         ("failed", self.failed)):
            if n:
                bits.append(f"{n} {label}")
        if self.cancelled:
            bits.append("stopped early")
        return "; ".join(bits)


def _source_image_dirs(conn) -> dict[int, Path]:
    dirs: dict[int, Path] = {}
    for row in conn.execute("SELECT id, config FROM sources"):
        try:
            cfg = json.loads(row["config"]) if row["config"] else {}
        except (json.JSONDecodeError, TypeError):
            cfg = {}
        if isinstance(cfg, dict) and cfg.get("image_dir"):
            dirs[row["id"]] = Path(cfg["image_dir"])
    return dirs


def _find_image(dataset_root: Path, row, image_dirs: dict[int, Path],
                allow_preview: bool) -> tuple[Path | None, bool]:
    """(path, is_preview). The original image first: a preview is the
    VAE's 256px reconstruction, fine for a caption but not what the user
    supplied."""
    base = image_dirs.get(row["source_id"])
    if base is not None and row["source_path"]:
        candidate = base / row["source_path"]
        if candidate.is_file():
            return candidate, False
    if allow_preview and row["preview_path"]:
        candidate = dataset_root / row["preview_path"]
        if candidate.is_file():
            return candidate, True
    return None, False


def caption_dataset(dataset_root: Path, captioner: Captioner, *,
                    overwrite: bool = False, allow_preview: bool = True,
                    prefix: str = "", suffix: str = "", lowercase: bool = False,
                    limit: int | None = None,
                    should_cancel: Callable[[], bool] | None = None,
                    on_progress: Callable[[int, int], None] | None = None,
                    commit_every: int = 20) -> CaptionReport:
    """Caption the dataset's images and write the captions into its DB.

    Only rows with an empty caption are touched unless ``overwrite``, so a
    re-run resumes. Rows are committed every ``commit_every`` captions and on
    exit (including cancel and error), so stopping keeps the work done.
    ``limit`` caps how many images are captioned this call (a cheap trial).
    """
    from PIL import Image

    db_path = dataset_root / "metadata.db"
    ensure_v2(db_path)
    report = CaptionReport()
    with _connect(db_path) as conn:
        image_dirs = _source_image_dirs(conn)
        rows = conn.execute(
            "SELECT id, prompt, source_id, source_path, preview_path "
            "FROM trajectories ORDER BY id").fetchall()
        report.total = len(rows)
        pending = 0
        try:
            for done, row in enumerate(rows, 1):
                if should_cancel is not None and should_cancel():
                    report.cancelled = True
                    break
                if limit is not None and report.captioned >= limit:
                    break
                if (row["prompt"] or "").strip() and not overwrite:
                    report.skipped_has_caption += 1
                    continue
                path, is_preview = _find_image(dataset_root, row, image_dirs, allow_preview)
                if path is None:
                    report.missing_image += 1
                    continue
                try:
                    with Image.open(path) as img:
                        text = captioner.caption(img.convert("RGB"))
                except Exception as exc:  # one bad image must not end the run
                    report.failed += 1
                    if len(report.errors) < 10:
                        report.errors.append(f"{path.name}: {exc}")
                    continue
                text = clean_caption(text, prefix, suffix, lowercase)
                if not text:
                    report.empty_output += 1
                    continue
                conn.execute("UPDATE trajectories SET prompt = ? WHERE id = ?",
                             (text, row["id"]))
                report.captioned += 1
                report.used_preview += int(is_preview)
                pending += 1
                if pending >= commit_every:
                    conn.commit()
                    pending = 0
                if on_progress is not None:
                    on_progress(done, report.total)
        finally:
            conn.commit()
    return report
