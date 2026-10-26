"""Prompts: base captions plus an ordered list of stages, evaluated per epoch.

Base captions come from the library and are never rewritten by a stage. Every stage is a pure function
of (seed, stage position, epoch, image id, text), so the prompts of epoch 7 are the same whether the run
got there straight or resumed at 7. The table keeps the prompts of the current epoch; ``set_epoch``
(called by the batcher before an epoch starts) recomputes them and announces ``PromptsChanged`` when
any differ, which is how the conditioning cache learns what to warm and what to drop.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from collections.abc import Callable, Mapping

from ..graph.events import PromptsChanged
from .ids import image_of
from .library import split_tags


class Lifetime(Enum):
    """How long a prompt's text is likely to be worth remembering."""

    STABLE = "stable"      # a caption of the set, or one of very few variants: worth keeping across runs
    FLEETING = "fleeting"  # one draw from a large space (shuffles, pools): unlikely to come back


class Prompt(str):
    """Prompt text that says how long it is worth remembering. It is a ``str`` everywhere else, so every
    consumer that ignores lifetimes keeps working; a plain ``str`` counts as stable."""

    lifetime: Lifetime

    def __new__(cls, text: str, lifetime: Lifetime = Lifetime.STABLE) -> Prompt:
        obj = super().__new__(cls, text)
        obj.lifetime = lifetime
        return obj


def lifetime_of(prompt: str) -> Lifetime:
    return getattr(prompt, "lifetime", Lifetime.STABLE)


@dataclass(frozen=True)
class Context:
    """What a stage may look at besides its own text."""

    image_id: str
    epoch: int
    rng: random.Random
    distinct: tuple[str, ...]  # distinct non-empty base captions of the whole set


class Stage(Protocol):
    lifetime: Lifetime
    """FLEETING when the stage's outputs come from a space too large to repeat, STABLE when they come from a few."""

    def apply(self, text: str, ctx: Context) -> str: ...


def _trim(caption: str, dividers: str) -> str:
    return caption.strip().rstrip(dividers + " ")


@dataclass(frozen=True)
class CaptionPool:
    """Mix in captions of other images. ``captions`` joins whole captions; ``parts`` pools their tags."""

    n_min: int = 1
    n_max: int = 3
    mode: str = "captions"
    include_own: bool = True
    dividers: str = ".,"
    lifetime = Lifetime.FLEETING

    def __post_init__(self) -> None:
        if self.mode not in ("captions", "parts"):
            raise ValueError(f"CaptionPool mode must be 'captions' or 'parts', got {self.mode!r}")
        if self.n_min < 1 or self.n_max < self.n_min:
            raise ValueError(f"CaptionPool needs 1 <= n_min <= n_max, got {self.n_min}, {self.n_max}")
        if not self.dividers:
            raise ValueError("CaptionPool needs at least one divider character")

    def apply(self, text: str, ctx: Context) -> str:
        rng, n = ctx.rng, ctx.rng.randint(self.n_min, self.n_max)
        chosen = [text] if self.include_own and text.strip() else []
        others = [c for c in ctx.distinct if c != text]
        chosen += rng.sample(others, max(0, min(n - len(chosen), len(others))))
        if self.mode == "captions":
            return ". ".join(t for t in (_trim(c, self.dividers) for c in chosen) if t)
        split = re.compile("[" + re.escape(self.dividers) + "]")
        parts, seen = [], set()
        for caption in chosen:
            for part in (p.strip() for p in split.split(caption)):
                if part and part.lower() not in seen:
                    seen.add(part.lower())
                    parts.append(part)
        rng.shuffle(parts)
        return ", ".join(parts)


@dataclass(frozen=True)
class CaptionDropout:
    """With probability ``p`` replace the caption by another image's (``substitute``) or by nothing (``empty``)."""

    p: float = 0.1
    mode: str = "substitute"
    lifetime = Lifetime.STABLE  # the result is the caption itself, another caption of the set, or nothing

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"dropout p must be in [0, 1], got {self.p}")
        if self.mode not in ("substitute", "empty"):
            raise ValueError(f"dropout mode must be 'substitute' or 'empty', got {self.mode!r}")

    def apply(self, text: str, ctx: Context) -> str:
        if ctx.rng.random() >= self.p:
            return text
        if self.mode == "empty":
            return ""
        return ctx.rng.choice(ctx.distinct) if len(ctx.distinct) >= 2 else text


@dataclass(frozen=True)
class TagSwap:
    """Reduce the caption to one tag per epoch, after the first ``keep_first`` tags, which always stay.

    ``random`` picks a tag each epoch; ``walk`` takes the tags in order, one per epoch, wrapping around.
    """

    mode: str = "random"
    keep_first: int = 0
    lifetime = Lifetime.FLEETING  # one variant per tag of every caption: the set of variants is large

    def __post_init__(self) -> None:
        if self.mode not in ("random", "walk"):
            raise ValueError(f"TagSwap mode must be 'random' or 'walk', got {self.mode!r}")
        if self.keep_first < 0:
            raise ValueError("keep_first must not be negative")

    def apply(self, text: str, ctx: Context) -> str:
        tags = split_tags(text)
        kept, rest = tags[:self.keep_first], tags[self.keep_first:]
        if not rest:
            return text
        pick = ctx.rng.choice(rest) if self.mode == "random" else rest[ctx.epoch % len(rest)]
        return ", ".join(kept + [pick])


@dataclass(frozen=True)
class TagShuffle:
    """Shuffle tag order each epoch, optionally dropping each tag (after ``keep_first``) with probability ``drop``."""

    keep_first: int = 0
    drop: float = 0.0
    lifetime = Lifetime.FLEETING

    def __post_init__(self) -> None:
        if not 0.0 <= self.drop < 1.0:
            raise ValueError(f"tag drop must be in [0, 1), got {self.drop}")

    def apply(self, text: str, ctx: Context) -> str:
        tags = split_tags(text)
        kept, rest = tags[:self.keep_first], tags[self.keep_first:]
        ctx.rng.shuffle(rest)
        rest = [t for t in rest if ctx.rng.random() >= self.drop] or rest[:1]
        return ", ".join(kept + rest)


class PromptTable:
    """Prompts for the images of one set. Chain tables with ``with_stage``; each keeps the same base."""

    def __init__(self, base: Mapping[str, str], stages: tuple[Stage, ...] = (), *, seed: int = 0,
                 listener: Callable[[PromptsChanged], None] | None = None):
        self._base = dict(base)
        self.stages = tuple(stages)
        self.seed = seed
        self.listener = listener
        self.version = 0
        self.epoch = -1
        self._active: dict[str, str] = {}
        self._distinct = self._distinct_of(self._base)

    @classmethod
    def from_images(cls, images, channel: str = "text", *, seed: int = 0) -> PromptTable:
        return cls({i: images.caption(i, channel) for i in images.ids}, seed=seed)

    @staticmethod
    def _distinct_of(base: Mapping[str, str]) -> tuple[str, ...]:
        return tuple(sorted({t for t in base.values() if t.strip()}))

    def with_stage(self, stage: Stage) -> PromptTable:
        """A new table with ``stage`` appended; base captions and seed are shared, state is not."""
        return PromptTable(self._base, self.stages + (stage,), seed=self.seed)

    def __len__(self) -> int:
        return len(self._base)

    def _compute(self, image_id: str, epoch: int) -> Prompt:
        text, fleeting = self._base.get(image_id, ""), False
        for position, stage in enumerate(self.stages):
            rng = random.Random(f"{self.seed}:{position}:{epoch}:{image_id}")
            changed = stage.apply(text, Context(image_id, epoch, rng, self._distinct))
            fleeting = fleeting or (changed != text and getattr(stage, "lifetime", Lifetime.STABLE) is Lifetime.FLEETING)
            text = changed
        return Prompt(text, Lifetime.FLEETING if fleeting else Lifetime.STABLE)

    def set_epoch(self, epoch: int) -> int:
        """Recompute every prompt for ``epoch``; returns how many differ from the previous epoch's."""
        fresh = {i: self._compute(i, epoch) for i in self._base}
        changed = sum(1 for i, t in fresh.items() if self._active.get(i) != t)
        self._active, self.epoch = fresh, epoch
        if changed:
            self._announce(changed)
        return changed

    def update_base(self, changes: Mapping[str, str]) -> None:
        """Replace base captions (a caption edit or an auto-captioner finishing) and re-derive the affected prompts."""
        self._base.update(changes)
        self._distinct = self._distinct_of(self._base)
        epoch = max(self.epoch, 0)
        for image_id in changes:
            self._active[image_id] = self._compute(image_id, epoch)
        self._announce(len(changes))

    def _announce(self, changed: int) -> None:
        self.version += 1
        if self.listener:
            self.listener(PromptsChanged(self.version, changed, len(self._base)))

    def prompt_for(self, sample: str) -> str:
        if self.epoch < 0:
            raise RuntimeError("PromptTable.set_epoch() has not been called; the batcher does this before each epoch")
        return self._active[image_of(sample)]

    def snapshot(self) -> tuple[int, dict[str, str]]:
        """(version, prompts by image id) as of now; a pass reads from this copy."""
        return self.version, dict(self._active)

    def universe(self) -> set[str]:
        return set(self._active.values())

    def vocabulary(self) -> Counter:
        """How often each tag occurs in the base captions."""
        return Counter(t for text in self._base.values() for t in split_tags(text))

    def knows(self, image_id: str) -> bool:
        return image_id in self._base
