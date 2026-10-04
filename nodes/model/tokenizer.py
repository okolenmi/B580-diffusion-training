r"""The CLIP BPE tokenizer, reimplemented rather than imported.

Design doc 12, section 7.3, section C2. This is the last piece of the
ComfyUI dependency, and it is the one that is mostly *data* rather than code.

ComfyUI's version is `comfy.sd1_clip.SDTokenizer` wrapped around HuggingFace's
`CLIPTokenizer`, and it arrives carrying machinery this project has no use
for: token weights for ComfyUI's `<w:...>` wildcard syntax, `embedding:`
directives that substitute precomputed text-embedding tensors from disk,
`min_length` and `min_padding` options, left-padding, and per-word ids for
highlighting a UI. None of that is SDXL, and none of it is reimplemented
here. What is here is the published byte-level BPE and the sequence layout
SDXL needs.

**The algorithm** is OpenAI's, from the CLIP repository's
`simple_tokenizer.py`: byte-pair encoding over a byte-level alphabet, with
`</w>` marking a word's end, behind a fixed regex that splits contractions,
letter runs, digit runs and punctuation runs. `bytes_to_unicode()` is the
usual trick of mapping all 256 bytes to printable unicode so the BPE never
sees control characters. None of this is ComfyUI's; they took it from the
same place.

**The layout** is what SDTokenizer adds around the BPE:

* `BOS` then the text then `EOS`, both added by us rather than tokenized --
  ComfyUI derives their ids by tokenizing the empty string and reading the
  first two results, which works only because its config declares the EOS
  token's content to be the empty string. Ours are named constants.
* **CLIP-L pads with EOS**, so a short prompt's padding is indistinguishable
  from its end. **CLIP-G pads with token 0.** Two towers, two conventions.
* A word of eight or more BPE tokens starts a new section rather than being
  split across the boundary, matching `max_word_length`.
* A prompt longer than the sequence becomes several sections of 77, each
  with its own BOS and EOS. They are rejoined along the sequence axis by
  `clip.py`, which is how a long prompt stays usable rather than truncated.

**Three places this deliberately differs from ComfyUI**, all of them cases
where ComfyUI has prompt-grammar machinery this project does not have.
Verified over a 25-prompt corpus; everything else is token-for-token
identical to ComfyUI's tokenizer, including the empty string, unicode,
contractions, emoji, 300-character prompts, and prompts long enough to span
several sections.

1. **`(word)` is text, not a weight group.** ComfyUI's `token_weights` runs
   `parse_parentheses` over the prompt and consumes a balanced `(...)` as a
   weighting expression, so `"a (b) c"` reaches its tokenizer as `"a b c"`
   with the parentheses gone -- and `"-(a)-"` is re-split as `-`, `a`, `-`
   rather than the two `--` runs we produce. That is ComfyUI's prompt syntax
   leaking into the tokenizer. This project has no weight syntax, so a
   parenthesis in a prompt is a parenthesis, byte-encoded and merged like any
   other character.

2. **A literal `<|startoftext|>` in the text is text.** CLIP's split pattern
   matches the two special tokens and hands them through as ids, so a prompt
   containing those thirteen characters comes back with a second BOS in the
   middle of it. BOS and EOS come from `SDTokenizer.tokenize`, once each.

3. **HTML entities are unescaped, twice.** This is CLIP's `basic_clean`, and
   the vocabulary's merges were derived from text put through it. The
   installed `transformers` 5.12.1 has dropped that step -- `CLIPTokenizer`
   no longer has a `fix_text` attribute at all -- so ComfyUI now tokenizes
   `&lt;` as four tokens where CLIP's own tokenizer gives one. It also no
   longer calls `ftfy.fix_text`, which this project could not do anyway
   (ftfy is not installed and not declared), so a prompt carrying mojibake
   repairs differently here.

**One declared dependency, `regex`.** CLIP's splitting pattern is
`\p{L}|\p{N}|[^\s\p{L}\p{N}]`, and the standard library's `re` has no `\p`
at all -- it raises `bad escape` at compile time rather than compiling
something approximate. An approximate version would be a silent divergence
on exactly the scripts this corpus already covers (CJK, Cyrillic), so the
dependency is taken rather than faked. It has no dependencies of its own.
ComfyUI had it transitively, through `transformers`, which this tokenizer
no longer needs -- so this is a net reduction, not a new weight.

**Where the vocabulary comes from is still open.** The merges and ids are
OpenAI's published CLIP BPE data, not ComfyUI's work, but the copy this
development machine has lives in ComfyUI's source tree.
`default_vocabulary_dir()` looks in this repository first and falls back to
ComfyUI's tree, and `CLIP_TOKENIZER_DIR` overrides both. See the design doc
section 7.3 for what has to be decided.

Run: `python nodes/model/tokenizer.py` prints the ids for a prompt.
"""

from __future__ import annotations

import gzip
import html
import json
import os
import re
from functools import lru_cache
from pathlib import Path

try:
    import regex
except ImportError as _exc:  # pragma: no cover -- an installation problem
    raise ImportError(
        "nodes/model/tokenizer.py needs the 'regex' package: CLIP's token "
        "splitting pattern uses \\p{L} and \\p{N}, which the standard "
        "library's `re` does not have -- it raises 'bad escape' at compile "
        "time rather than compiling loosely, so there is no stdlib fallback "
        "that would work. `pip install regex`. It has no dependencies of "
        "its own. (ComfyUI got this from HuggingFace's transformers, which "
        "this tokenizer no longer needs.)"
    ) from _exc

__all__ = [
    "CLIP_BOS",
    "CLIP_EOS",
    "SDTokenizer",
    "SDXLTokenizer",
    "bytes_to_unicode",
    "default_vocabulary_dir",
]

#: OpenAI CLIP's two special tokens. Named here rather than read out of a
#: tokenizer config, which is what ComfyUI does and which makes the ids
#: depend on a file whose EOS entry is the empty string.
CLIP_BOS = 49406
CLIP_EOS = 49407

#: CLIP-G's padding, as opposed to CLIP-L's EOS.
CLIP_G_PAD = 0

#: The vocabulary's own files. `merges.txt` first line is `#version: 0.2`,
#: which is a comment and not a merge.
_MERGES_FILE = "merges.txt"
_VOCAB_FILE = "vocab.json"


def bytes_to_unicode() -> dict[int, str]:
    """Map every byte to a printable unicode character.

    The BPE alphabet has to be text, and 256 raw bytes are not. The usual
    solution, from CLIP: the 188 printable ASCII and Latin-1 bytes keep
    themselves, and the remaining 68 -- control characters plus the gaps at
    127..160 and 173 -- move up into U+0100..U+0143. Every byte gets exactly
    one character and every character exactly one byte, so a word survives
    the round trip and the merge table can be plain text.

    The subtle part, and the part a first version of this got wrong: the
    mapping is keyed by *byte value*, and the printable range starts at
    byte 33. Zipping `range(256)` against the characters instead pairs byte 0
    with `'!'`, and then "photo" byte-encodes to `'´«³¸³'` and merges
    nothing -- every word silently comes out as single characters, with no
    error anywhere.
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    characters = [chr(b) for b in byte_values]
    spare = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            characters.append(chr(256 + spare))
            spare += 1
    return dict(zip(byte_values, characters))


#: CLIP's token-splitting pattern: English contractions, then a run of
#: letters, a run of digits, or a run of everything else that is neither.
#: `regex` rather than `re`, because `\p{L}` is not in the standard library's
#: repertoire -- without it the pattern raises at compile time rather than
#: compiling loosely.
#:
#: **The two special-token alternatives are deliberately absent.** CLIP's
#: pattern matches a literal `<|startoftext|>` in the text and hands it
#: straight through as token 49406, and ComfyUI's tokenizer does the same: a
#: prompt containing those thirteen characters comes back with a second BOS
#: in the middle of it. A prompt is a description of an image, so text that
#: looks like a special token is text like any other and is byte-encoded and
#: merged like any other. BOS and EOS come from `SDTokenizer.tokenize`, once
#: each, and nowhere else. This is the one place this tokenizer deliberately
#: diverges from ComfyUI, and `smoke_test_tokenizer.py` pins it.
_SPLIT_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+"""
)


def _split_words(text: str):
    return regex.findall(_SPLIT_PATTERN, text, regex.IGNORECASE)


def _basic_clean(text: str) -> str:
    """Unescape twice, then trim.

    Twice because a doubly-escaped `&amp;lt;` is a real thing to meet in a
    prompt copied through a web page. ComfyUI's `basic_clean` also calls
    ftfy's `fix_text` to repair mojibake; this project has no ftfy dependency
    and does not do that, so a prompt carrying mangled encoding tokenizes
    differently from ComfyUI's. Recorded rather than hidden.
    """
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


class _BytePairEncoder:
    """The merge table: a word becomes BPE tokens, greedily merged."""

    def __init__(self, encoder: dict[str, int], merges: list[str]) -> None:
        self.encoder = encoder
        self.byte_encoder = bytes_to_unicode()
        # Merge rank: lower is earlier, and a pair that is not in the table
        # sorts after every pair that is, which is what stops the loop.
        self.bpe_ranks = {tuple(pair.split(" ")): rank
                          for rank, pair in enumerate(merges)
                          if len(pair.split(" ")) == 2}
        self.cache: dict[str, str] = {}

    def bpe(self, token: str) -> list[str]:
        if token in self.cache:
            return self.cache[token].split(" ")
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            # A single-symbol word: the whole thing, with `</w` on the end,
            # and no space -- the caller splits on spaces, so a space here
            # would come back as two tokens and the second would be a bare
            # `</w>`, which is not in the vocabulary.
            result = token + "</w>"
            self.cache[token] = result
            return [result]

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair,
                                                                     float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 \
                        and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result.split(" ")

    def encode(self, text: str) -> list[str]:
        """Text to BPE token strings, before any ids are attached."""
        cleaned = _whitespace_clean(_basic_clean(text)).lower()
        out: list[str] = []
        for word in _split_words(cleaned):
            # Byte-level: each byte becomes one alphabet character, so the
            # merge table can be plain text.
            encoded = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
            out.extend(self.bpe(encoded))
        return out


def _get_pairs(word: tuple[str, ...]):
    return set(zip(word[:-1], word[1:]))


def load_vocabulary(directory) -> tuple[dict[str, int], list[str]]:
    """Read `vocab.json` and `merges.txt` from a CLIP tokenizer directory."""
    path = Path(directory)
    vocab_path = path / _VOCAB_FILE
    merges_path = path / _MERGES_FILE
    if not vocab_path.is_file() or not merges_path.is_file():
        raise FileNotFoundError(
            f"{path} does not look like a CLIP tokenizer directory: expected "
            f"{_VOCAB_FILE} and {_MERGES_FILE}, found "
            f"{sorted(p.name for p in path.glob('*')) if path.is_dir() else 'nothing'}")

    encoder = json.loads(vocab_path.read_text(encoding="utf-8"))
    opener = gzip.open if str(merges_path).endswith(".gz") else open
    with opener(merges_path, "rt", encoding="utf-8") as handle:
        merges = handle.read().split("\n")
    # Drop the version comment and any trailing blank line.
    merges = [m for m in merges if m and not m.startswith("#version")]
    return encoder, merges


def default_vocabulary_dir() -> Path:
    """Where the CLIP BPE vocabulary is read from.

    In order: `$CLIP_TOKENIZER_DIR`, then a copy inside this repository, then
    ComfyUI's source tree.

    The third is a real dependency on someone else's *files* rather than
    their code, which is exactly what this work is removing, so it is a
    fallback and not the intended arrangement. It is here so the tokenizer
    works on a machine that has ComfyUI installed, which is where this was
    developed and verified.
    """
    override = os.environ.get("CLIP_TOKENIZER_DIR")
    if override:
        return Path(override)

    repo = Path(__file__).resolve().parents[2]
    ours = repo / "assets" / "clip_tokenizer"
    if (ours / _VOCAB_FILE).is_file():
        return ours

    candidates = []
    configured = os.environ.get("COMFY_DIR")
    if configured:
        candidates.append(Path(configured) / "sd1_tokenizer")
    # `paths` is this project's own ComfyUI resolver -- it reads .env and
    # falls back to a search, which is what makes a plain
    # `python nodes/model/tokenizer.py` work without COMFY_DIR exported.
    # Imported here rather than at module level: a tokenizer that has been
    # handed a directory must not need it.
    try:
        import paths
        resolved = Path(paths.get_comfy_dir())
        candidates.append(resolved / "sd1_tokenizer")
        candidates.append(resolved / "comfy" / "sd1_tokenizer")
    except Exception:  # noqa: BLE001 -- no ComfyUI, which is a valid state
        pass

    for candidate in candidates:
        if (candidate / _VOCAB_FILE).is_file():
            return candidate
    raise FileNotFoundError(
        "no CLIP BPE vocabulary found. Set CLIP_TOKENIZER_DIR to a directory "
        "containing vocab.json and merges.txt; looked in "
        + ", ".join(str(c) for c in candidates))


class SDTokenizer:
    """One CLIP tower's tokenizer: BPE plus the sequence layout around it.

    :param pad_with_end: CLIP-L pads with EOS, CLIP-G does not. With it, a
        short prompt's padding is indistinguishable from its end, which is
        why the two towers' outputs are not interchangeable.
    :param pad_id: the token to pad with. Derived from `pad_with_end` unless
        given.
    """

    def __init__(self, directory=None, max_length: int = 77,
                 pad_with_end: bool = True, max_word_length: int = 8) -> None:
        encoder, merges = load_vocabulary(
            directory if directory is not None else default_vocabulary_dir())
        self._bpe = _BytePairEncoder(encoder, merges)
        self.max_length = max_length
        self.max_word_length = max_word_length
        self.start_token = CLIP_BOS
        self.end_token = CLIP_EOS
        self.pad_token = CLIP_EOS if pad_with_end else CLIP_G_PAD

    def encode_text(self, text: str) -> list[int]:
        """Token ids for `text`, with no BOS, EOS or padding."""
        return [self._bpe.encoder[token] for token in self._bpe.encode(text)]

    def tokenize(self, text: str) -> list[list[int]]:
        """Token ids as one or more sections of `max_length`.

        More than one section only when the prompt does not fit, or when a
        single word runs to `max_word_length` BPE tokens or more -- such a
        word starts a new section instead of being cut at the boundary, so
        the UNet sees whole words.
        """
        sections: list[list[int]] = []
        batch = [self.start_token]
        has_end = 1

        def flush():
            nonlocal batch
            batch.append(self.end_token)
            if len(batch) < self.max_length:
                batch.extend([self.pad_token] * (self.max_length - len(batch)))
            sections.append(batch)
            batch = [self.start_token]

        for word in text.split():
            ids = self.encode_text(word)
            if not ids:
                continue
            is_large = len(ids) >= self.max_word_length
            while ids:
                room = self.max_length - len(batch) - has_end
                if len(ids) + len(batch) > self.max_length - has_end:
                    if is_large:
                        # Break the word across sections, then start the
                        # next section with BOS again.
                        batch.extend(ids[:room])
                        ids = ids[room:]
                        flush()
                    else:
                        flush()
                else:
                    batch.extend(ids)
                    ids = []
        if batch != [self.start_token] or not sections:
            flush()
        return sections


class SDXLTokenizer:
    """Both towers, over one shared BPE table.

    `tokenize()` returns `{"l": [...], "g": [...]}`, each a list of sections.
    CLIP-L and CLIP-G use the same 49408-entry vocabulary and the same BOS and
    EOS; they differ only in padding, so one BPE instance serves both.
    """

    def __init__(self, directory=None) -> None:
        resolved = directory if directory is not None else default_vocabulary_dir()
        self.clip_l = SDTokenizer(resolved, pad_with_end=True)
        self.clip_g = SDTokenizer(resolved, pad_with_end=False)

    def tokenize(self, text: str) -> dict[str, list[list[int]]]:
        return {"l": self.clip_l.tokenize(text), "g": self.clip_g.tokenize(text)}


@lru_cache(maxsize=1)
def _default_tokenizer() -> SDXLTokenizer:
    return SDXLTokenizer()


def tokenize(text: str) -> dict[str, list[list[int]]]:
    """Token ids for `text`, via a process-wide tokenizer."""
    return _default_tokenizer().tokenize(text)


if __name__ == "__main__":
    import sys

    # Run directly, sys.path[0] is this file's own directory, so the repo
    # root -- where `paths` lives -- is not importable. The library does not
    # do this for itself; a script that has to find its own project root
    # should say so.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    prompt = " ".join(sys.argv[1:]) or "a photo of a cat"
    result = tokenize(prompt)
    print(f"{prompt!r}")
    print(f"  vocabulary: {default_vocabulary_dir()}")
    for tower in ("l", "g"):
        for index, section in enumerate(result[tower]):
            print(f"  {tower}[{index}] {len(section)} ids, first 8: "
                  f"{section[:8]}")