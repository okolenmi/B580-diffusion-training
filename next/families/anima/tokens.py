r"""Anima's two tokenizers: Qwen's byte-level BPE (for the Qwen3 encoder) and T5's Unigram (ids for the text adapter).

Both read the whole prompt as plain text: no emphasis syntax, parentheses are characters (booru tags carry them), and no
padding. Qwen gets no start or end token, an empty prompt becomes one pad token; T5 gets ``</s>`` at the end.

Needs the Hugging Face ``tokenizers`` package (ComfyUI installs it). The vocabularies are not stored in the code tree;
``vocabulary_dir`` says where each is looked for.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

QWEN_PAD = 151643
T5_END = 1
# Qwen2's pre-tokenizer pattern: contractions, letter runs, single digits, punctuation runs, line breaks, spaces.
QWEN_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

_KINDS = {"qwen": ("QWEN_TOKENIZER_DIR", "qwen_tokenizer", "qwen25_tokenizer", "vocab.json"),
          "t5": ("T5_TOKENIZER_DIR", "t5_tokenizer", "t5_tokenizer", "tokenizer.json")}


def vocabulary_dir(kind: str, extra: tuple[str | os.PathLike, ...] = ()) -> Path:
    """First folder holding the files of ``kind`` ("qwen": ``vocab.json``, ``merges.txt``, ``tokenizer_config.json``; "t5":
    ``tokenizer.json``): the ``*_TOKENIZER_DIR`` variable, then ``extra``, then ``<repo>/assets/<kind>_tokenizer``, then the
    folders of a ComfyUI checkout (``$COMFY_DIR/comfy/text_encoders``)."""
    variable, asset, comfy, marker = _KINDS[kind]
    candidates: list[Path] = []
    if os.environ.get(variable):
        candidates.append(Path(os.environ[variable]))
    candidates.extend(Path(p) for p in extra)
    candidates.append(Path(__file__).resolve().parents[3] / "assets" / asset)
    if os.environ.get("COMFY_DIR"):
        candidates.append(Path(os.environ["COMFY_DIR"]) / "comfy" / "text_encoders" / comfy)
    for candidate in candidates:
        if (candidate / marker).is_file():
            return candidate
    raise FileNotFoundError(f"no {kind} tokenizer files found; set {variable} to a folder holding {marker}"
                            f" (searched: {', '.join(str(c) for c in candidates)})")


def _library():
    try:
        import tokenizers
    except ImportError as exc:  # pragma: no cover
        raise ImportError("the Anima tokenizers need the 'tokenizers' package (pip install tokenizers)") from exc
    return tokenizers


class QwenTokenizer:
    def __init__(self, directory: str | os.PathLike | None = None):
        lib = _library()
        folder = Path(directory) if directory is not None else vocabulary_dir("qwen")
        vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        lines = (folder / "merges.txt").read_text(encoding="utf-8").splitlines()
        merges = [tuple(line.split(" ")) for line in lines if line and not line.startswith("#version")]
        tok = lib.Tokenizer(lib.models.BPE(vocab, merges))
        tok.normalizer = lib.normalizers.NFC()
        tok.pre_tokenizer = lib.pre_tokenizers.Sequence([
            lib.pre_tokenizers.Split(lib.Regex(QWEN_PATTERN), behavior="isolated"),
            lib.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tok.decoder = lib.decoders.ByteLevel()
        config = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
        specials = [lib.AddedToken(v["content"], special=True, normalized=False)
                    for _, v in sorted(config.get("added_tokens_decoder", {}).items(), key=lambda kv: int(kv[0]))]
        tok.add_special_tokens(specials)
        self._tok = tok

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids or [QWEN_PAD]

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)


class T5Tokenizer:
    def __init__(self, directory: str | os.PathLike | None = None):
        folder = Path(directory) if directory is not None else vocabulary_dir("t5")
        self._tok = _library().Tokenizer.from_file(str(folder / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        """The ids of ``text`` followed by ``</s>``."""
        ids = self._tok.encode(text).ids
        return ids if ids and ids[-1] == T5_END else ids + [T5_END]


class AnimaTokenizer:
    def __init__(self, qwen_dir=None, t5_dir=None):
        self.qwen, self.t5 = QwenTokenizer(qwen_dir), T5Tokenizer(t5_dir)

    def tokenize(self, text: str) -> tuple[list[int], list[int]]:
        return self.qwen.encode(text), self.t5.encode(text)
