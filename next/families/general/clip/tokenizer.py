r"""The CLIP byte-level BPE tokenizer and the sequence layout around it (from OpenAI's published algorithm).

Layout: ``BOS`` + text + ``EOS``, padded to ``max_length``. CLIP-L pads with EOS and CLIP-G with
token 0. A prompt too long for one section becomes several sections, each with its own BOS and EOS,
and a word of ``max_word_length`` or more BPE tokens starts a new section instead of being cut.

Prompt text is only text: parentheses are characters, not weights, and ``<|startoftext|>`` typed in
a prompt is not a special token. HTML entities are unescaped twice, as CLIP's own cleaning does.
Mojibake repair (``ftfy``) is not done.

Needs the ``regex`` package: the splitting pattern uses ``\p{L}`` and ``\p{N}``, which ``re`` rejects.
The vocabulary (``vocab.json`` and ``merges.txt``) is OpenAI's data and is not stored in the code tree;
``default_vocabulary_dir`` says where it is looked for.
"""

from __future__ import annotations

import gzip
import html
import json
import os
import re
from pathlib import Path

try:
    import regex
except ImportError as exc:  # pragma: no cover
    raise ImportError("the CLIP tokenizer needs the 'regex' package (pip install regex)") from exc

CLIP_BOS = 49406
CLIP_EOS = 49407
CLIP_G_PAD = 0
VOCAB_FILE = "vocab.json"
MERGES_FILE = "merges.txt"

_SPLIT = r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""


def bytes_to_unicode() -> dict[int, str]:
    """Every byte to one printable character. Printable bytes keep themselves; the rest move to U+0100 up.

    Keyed by byte value: pairing ``range(256)`` with the printable list instead maps byte 0 to '!' and
    every word silently fails to merge.
    """
    values = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
              + list(range(ord("\xae"), ord("\xff") + 1)))
    chars = [chr(b) for b in values]
    spare = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            chars.append(chr(256 + spare))
            spare += 1
    return dict(zip(values, chars))


def _clean(text: str) -> str:
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip().lower()


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word[:-1], word[1:]))


class BytePairEncoder:
    def __init__(self, encoder: dict[str, int], merges: list[str]) -> None:
        self.encoder = encoder
        self._bytes = bytes_to_unicode()
        self._ranks = {tuple(m.split(" ")): i for i, m in enumerate(merges) if len(m.split(" ")) == 2}
        self._cache: dict[str, list[str]] = {}

    def _merge(self, token: str) -> list[str]:
        if token in self._cache:
            return self._cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            self._cache[token] = [token + "</w>"]
            return self._cache[token]
        while True:
            bigram = min(pairs, key=lambda p: self._ranks.get(p, float("inf")))
            if bigram not in self._ranks:
                break
            first, second = bigram
            merged: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    merged.extend(word[i:])
                    break
                merged.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    merged.append(first + second)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        self._cache[token] = list(word)
        return self._cache[token]

    def tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for word in regex.findall(_SPLIT, _clean(text), regex.IGNORECASE):
            out.extend(self._merge("".join(self._bytes[b] for b in word.encode("utf-8"))))
        return out


def load_vocabulary(directory: str | os.PathLike) -> tuple[dict[str, int], list[str]]:
    path = Path(directory)
    vocab, merges = path / VOCAB_FILE, path / MERGES_FILE
    if not vocab.is_file() or not merges.is_file():
        found = sorted(p.name for p in path.glob("*")) if path.is_dir() else "nothing"
        raise FileNotFoundError(f"{path} is not a CLIP tokenizer folder: expected {VOCAB_FILE} and {MERGES_FILE}, found {found}")
    encoder = json.loads(vocab.read_text(encoding="utf-8"))
    opener = gzip.open if str(merges).endswith(".gz") else open
    with opener(merges, "rt", encoding="utf-8") as handle:
        lines = handle.read().split("\n")
    return encoder, [m for m in lines if m and not m.startswith("#version")]


def default_vocabulary_dir(extra: tuple[str | os.PathLike, ...] = ()) -> Path:
    """First folder holding the vocabulary: ``$CLIP_TOKENIZER_DIR``, then ``extra``, then
    ``<repo>/assets/clip_tokenizer``, then ``$COMFY_DIR/sd1_tokenizer``."""
    candidates: list[Path] = []
    if os.environ.get("CLIP_TOKENIZER_DIR"):
        candidates.append(Path(os.environ["CLIP_TOKENIZER_DIR"]))
    candidates.extend(Path(p) for p in extra)
    candidates.append(Path(__file__).resolve().parents[4] / "assets" / "clip_tokenizer")
    if os.environ.get("COMFY_DIR"):
        candidates.append(Path(os.environ["COMFY_DIR"]) / "sd1_tokenizer")
    for candidate in candidates:
        if (candidate / VOCAB_FILE).is_file():
            return candidate
    raise FileNotFoundError("no CLIP vocabulary found; set CLIP_TOKENIZER_DIR to a folder holding "
                            f"{VOCAB_FILE} and {MERGES_FILE}. Looked in: " + ", ".join(map(str, candidates)))


class SectionTokenizer:
    """One tower's tokenizer: BPE plus BOS/EOS/padding layout."""

    def __init__(self, bpe: BytePairEncoder, *, pad_with_end: bool, max_length: int = 77, max_word_length: int = 8):
        self._bpe = bpe
        self.max_length, self.max_word_length = max_length, max_word_length
        self.start, self.end = CLIP_BOS, CLIP_EOS
        self.pad = CLIP_EOS if pad_with_end else CLIP_G_PAD

    def encode_text(self, text: str) -> list[int]:
        return [self._bpe.encoder[t] for t in self._bpe.tokens(text)]

    def tokenize(self, text: str) -> list[list[int]]:
        """One or more sections of exactly ``max_length`` ids."""
        sections: list[list[int]] = []
        batch = [self.start]

        def flush() -> None:
            nonlocal batch
            batch.append(self.end)
            batch.extend([self.pad] * (self.max_length - len(batch)))
            sections.append(batch)
            batch = [self.start]

        for word in text.split():
            ids = self.encode_text(word)
            large = len(ids) >= self.max_word_length
            while ids:
                room = self.max_length - len(batch) - 1
                if len(ids) + len(batch) > self.max_length - 1:
                    if large:
                        batch.extend(ids[:room])
                        ids = ids[room:]
                    flush()
                else:
                    batch.extend(ids)
                    ids = []
        if batch != [self.start] or not sections:
            flush()
        return sections


class CLIPTokenizer:
    """Both towers over one shared table. ``tokenize`` returns ``{"l": sections, "g": sections}``."""

    def __init__(self, directory: str | os.PathLike | None = None) -> None:
        encoder, merges = load_vocabulary(directory if directory is not None else default_vocabulary_dir())
        bpe = BytePairEncoder(encoder, merges)
        self.l = SectionTokenizer(bpe, pad_with_end=True)
        self.g = SectionTokenizer(bpe, pad_with_end=False)

    def tokenize(self, text: str) -> dict[str, list[list[int]]]:
        return {"l": self.l.tokenize(text), "g": self.g.tokenize(text)}
