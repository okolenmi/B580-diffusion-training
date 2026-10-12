"""CaptionPool: a prompt for every image drawn from the dataset's real captions.

The problem it answers. A dataset whose captions are all empty -- or all one
string -- presents a constant text conditioning at every step; cross-attention's
K/V projections then get no signal to be shaped by (manager/loader.py's
caption_dropout note, archive/bad-LoRA-investigation/MEASURED-captionless-
conditioning-break.md). Caption dropout swaps in *one* other caption with some
probability. This builds a richer prompt per image out of *several* captions the
dataset really has, so the run sees varied but in-distribution conditioning on
every sample and nothing foreign enters the tokenizer.

What a prompt is made of. For an image, ``n`` is drawn from ``n_min..n_max`` and
that many captions are taken from the distinct captions of the dataset (the
image's own caption first when ``include_own`` and it has one):

* ``mode="captions"``: the chosen captions, each trimmed of trailing dividers,
  joined with ``". "`` -- ``.`` is the divider *between* captions.
* ``mode="parts"``: every chosen caption is split on the dividers (``.`` and
  ``,``) into parts -- its tags, its "classes" -- the parts are de-duplicated,
  shuffled, and joined with ``", "``. A prompt is then a random bag of the
  dataset's own tags.

Re-randomizing vs cached. ``variants=K > 0`` (default 4) draws K prompts per
image **once**, up front (seeded); every later epoch picks one of the K. The set
of prompts is therefore finite and known (``universe()``), which is what lets
the text-encoder prewarm encode all of them and unload CLIP -- prewarm only
pays off with a bounded prompt set. ``variants=0`` draws a *fresh* prompt every
time the image is served: maximal variety, no bounded set, so no prewarm (each
new prompt is a cache miss that re-loads CLIP).

CLIP reads 77 tokens (the rest is cut). Many captions joined together can
overflow it; ``"parts"`` mode drops random parts when it does, ``"captions"``
mode drops the tail, which is why the image's own caption goes first.

No torch, no nodes/ imports -- manager/ must not depend on nodes/.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

MODES = ("captions", "parts")


class CaptionPool:

    def __init__(self, n_min: int = 1, n_max: int = 3, mode: str = "captions",
                 include_own: bool = True, variants: int = 4,
                 seed: int | None = None, dividers: str = ".,"):
        if mode not in MODES:
            raise ValueError(f"CaptionPool mode must be one of {MODES}, got {mode!r}")
        n_min, n_max, variants = int(n_min), int(n_max), int(variants)
        if n_min < 1 or n_max < n_min:
            raise ValueError(
                f"CaptionPool needs 1 <= n_min <= n_max, got n_min={n_min}, "
                f"n_max={n_max}")
        if variants < 0:
            raise ValueError(f"CaptionPool variants must be >= 0, got {variants}")
        if not dividers:
            raise ValueError("CaptionPool needs at least one divider character")
        self.n_min = n_min
        self.n_max = n_max
        self.mode = mode
        self.include_own = bool(include_own)
        self.variants = variants
        self.seed = seed
        self.dividers = dividers
        self._split = re.compile("[" + re.escape(dividers) + "]")
        self._rng = random.Random(seed)
        self._own: list[str] = []
        self._distinct: list[str] = []
        self._cached: list[list[str]] = []
        self.active = False

    # -- preparation ------------------------------------------------------

    def prepare(self, own_captions: Sequence[str]) -> bool:
        """Index the dataset. Returns False (and stays inactive, so every
        image keeps its own caption) when there is nothing to draw from: a
        pool of fewer than two distinct non-empty captions cannot vary."""
        self._own = list(own_captions)
        self._distinct = sorted({c for c in self._own if c and c.strip()})
        self.active = len(self._distinct) >= 2
        self._rng = random.Random(self.seed)
        self._cached = []
        if self.active and self.variants > 0:
            self._cached = [
                self._unique([self._compose(i) for _ in range(self.variants)])
                for i in range(len(self._own))]
        return self.active

    def _unique(self, prompts: list[str]) -> list[str]:
        return list(dict.fromkeys(prompts))

    # -- drawing ----------------------------------------------------------

    def prompt_for(self, index: int) -> str:
        """The prompt to train image ``index`` on right now."""
        if not self.active:
            return self._own[index]
        if self.variants > 0:
            return self._rng.choice(self._cached[index])
        return self._compose(index)

    def universe(self) -> list[str]:
        """Every prompt this pool can serve; empty when it is not enumerable
        (``variants=0``) or inactive. The prewarm warms exactly this set."""
        if not self.active or self.variants == 0:
            return []
        return sorted({p for variants in self._cached for p in variants})

    def _compose(self, index: int) -> str:
        n = self._rng.randint(self.n_min, self.n_max)
        own = self._own[index]
        chosen: list[str] = []
        if self.include_own and own and own.strip():
            chosen.append(own)
        others = [c for c in self._distinct if c not in chosen]
        want = max(0, min(n - len(chosen), len(others)))
        chosen += self._rng.sample(others, want)
        if self.mode == "captions":
            return ". ".join(self._trim(c) for c in chosen if self._trim(c))
        parts: list[str] = []
        seen: set[str] = set()
        for caption in chosen:
            for part in self._split.split(caption):
                part = part.strip()
                if part and part.lower() not in seen:
                    seen.add(part.lower())
                    parts.append(part)
        self._rng.shuffle(parts)
        return ", ".join(parts)

    def _trim(self, caption: str) -> str:
        return caption.strip().rstrip(self.dividers + " ")
