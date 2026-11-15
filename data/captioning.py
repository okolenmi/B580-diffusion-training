"""Captions for images that have none: a caption model run over a set, written into the library.

``Captioner`` is the one interface a caption model implements; backends are registered by name, so another model
(a local tagger, an API) is one registered factory and nothing else here knows which one wrote a caption.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Callable

from .library import ImageLibrary, ImageSet


class Captioner(ABC):
    @abstractmethod
    def caption(self, image) -> str:
        """A caption for one RGB PIL image; empty when the model said nothing."""

    def close(self) -> None:
        """Release the model."""


CAPTIONERS: dict[str, Callable[..., Captioner]] = {}


def register_captioner(name: str, factory: Callable[..., Captioner]) -> None:
    """``factory(model_id=, device=, max_new_tokens=)`` returns a Captioner and may ignore what means nothing to it."""
    CAPTIONERS[name] = factory


def make_captioner(backend: str, **options) -> Captioner:
    if backend not in CAPTIONERS:
        raise ValueError(f"unknown captioner backend {backend!r}; registered: {sorted(CAPTIONERS)}")
    return CAPTIONERS[backend](**options)


class TransformersCaptioner(Captioner):
    """Any Hugging Face ``image-to-text`` model (BLIP, GIT, ...) run locally; transformers is imported only when used."""

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base", device: str = "cpu", max_new_tokens: int = 40, **_ignored):
        try:
            from transformers import pipeline
        except ImportError as error:
            raise RuntimeError(f"the 'transformers' captioner needs the transformers package (pip install transformers): {error}") from error
        self._tokens = int(max_new_tokens)
        self._pipe = pipeline("image-to-text", model=model_id, device=device)

    def caption(self, image) -> str:
        out = self._pipe(image, max_new_tokens=self._tokens)
        return (out[0].get("generated_text") or "").strip() if out else ""

    def close(self) -> None:
        self._pipe = None


register_captioner("transformers", TransformersCaptioner)


def clean_caption(text: str, prefix: str = "", suffix: str = "", lowercase: bool = False) -> str:
    """Whitespace collapsed, a trailing divider dropped, the model's text optionally lowercased, trigger words around it.
    Nothing from the model stays empty: a bare prefix is not a caption."""
    text = " ".join((text or "").split()).rstrip(" .,")
    if not text:
        return ""
    if lowercase:
        text = text.lower()
    return ", ".join(p.strip().rstrip(" ,") for p in (prefix, text, suffix) if p and p.strip())


@dataclass
class CaptionReport:
    total: int = 0
    captioned: int = 0
    kept: int = 0
    unreadable: int = 0
    empty: int = 0
    cancelled: bool = False
    written: dict[str, str] = field(default_factory=dict)  # image id -> caption
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        bits = [f"{self.captioned} captioned of {self.total}"]
        bits += [f"{n} {label}" for label, n in (("kept as they were", self.kept), ("model said nothing", self.empty),
                                                  ("unreadable", self.unreadable)) if n]
        return "; ".join(bits + (["stopped early"] if self.cancelled else []))


def caption_images(library: ImageLibrary, images: ImageSet, captioner: Captioner, *, channel: str = "text", overwrite: bool = False,
                   prefix: str = "", suffix: str = "", lowercase: bool = False, limit: int | None = None,
                   should_stop: Callable[[], bool] | None = None, progress: Callable[[int, int], None] | None = None,
                   commit_every: int = 20) -> CaptionReport:
    """Caption the images of ``images`` into ``channel``. Only empty captions are written unless ``overwrite``, so a rerun
    resumes; what is done is committed as it goes, so stopping keeps it."""
    from PIL import Image

    report = CaptionReport(total=len(images))
    pending = 0
    try:
        for done, image_id in enumerate(images.ids, 1):
            if should_stop is not None and should_stop():
                report.cancelled = True
                break
            if limit is not None and report.captioned >= limit:
                break
            if images.caption(image_id, channel).strip() and not overwrite:
                report.kept += 1
            else:
                try:
                    with Image.open(images.paths[image_id]) as file:
                        text = clean_caption(captioner.caption(file.convert("RGB")), prefix, suffix, lowercase)
                except Exception as error:  # one bad image must not end the run
                    report.unreadable += 1
                    if len(report.errors) < 10:
                        report.errors.append(f"{images.paths[image_id]}: {error}")
                else:
                    if text:
                        library.set_caption(image_id, text, channel, commit=False)
                        report.written[image_id] = text
                        report.captioned += 1
                        pending += 1
                        if pending >= commit_every:
                            library.commit()
                            pending = 0
                    else:
                        report.empty += 1
            if progress:
                progress(done, len(images))
    finally:
        library.commit()
    return report
