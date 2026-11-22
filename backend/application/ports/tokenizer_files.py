"""The vocabulary files the prompt tokenizers read, kept in ``<project>/assets/<set>_tokenizer``.

They are published data (OpenAI's CLIP vocabulary, Qwen's and T5's tokenizer files), not code, so they are downloaded once from the
model hubs instead of being found inside someone else's application. The trainer's loaders check this folder before anything else.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..errors import ApplicationError


class TokenizerDownloadError(ApplicationError):
    code = "tokenizer_download_failed"
    status_code = 502


@dataclass(frozen=True, slots=True)
class TokenizerSet:
    key: str
    label: str
    needed_for: str
    #: file name -> download URLs, tried in order
    files: dict[str, tuple[str, ...]]


_HUB = "https://huggingface.co"
SETS: tuple[TokenizerSet, ...] = (
    TokenizerSet("clip", "CLIP vocabulary", "SDXL prompts", {
        "vocab.json": (f"{_HUB}/openai/clip-vit-large-patch14/resolve/main/vocab.json",),
        "merges.txt": (f"{_HUB}/openai/clip-vit-large-patch14/resolve/main/merges.txt",),
    }),
    TokenizerSet("qwen", "Qwen tokenizer", "Anima prompts", {
        "vocab.json": (f"{_HUB}/Qwen/Qwen2.5-0.5B/resolve/main/vocab.json",),
        "merges.txt": (f"{_HUB}/Qwen/Qwen2.5-0.5B/resolve/main/merges.txt",),
        "tokenizer_config.json": (f"{_HUB}/Qwen/Qwen2.5-0.5B/resolve/main/tokenizer_config.json",),
    }),
    TokenizerSet("t5", "T5 tokenizer", "Anima prompts", {
        "tokenizer.json": (f"{_HUB}/google/t5-v1_1-xxl/resolve/main/tokenizer.json",
                           f"{_HUB}/google-t5/t5-small/resolve/main/tokenizer.json"),
    }),
)


def set_by_key(key: str) -> TokenizerSet:
    for item in SETS:
        if item.key == key:
            return item
    raise TokenizerDownloadError(f"'{key}' is not a tokenizer set; known: {', '.join(s.key for s in SETS)}")


def folder_of(project_root: Path, key: str) -> Path:
    return project_root / "assets" / f"{key}_tokenizer"


def check_content(name: str, data: bytes) -> str | None:
    """Why ``data`` cannot be the file ``name``, or None. Catches an error page saved in place of the file."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return "not text"
    if name.endswith(".json"):
        try:
            parsed = json.loads(text)
        except ValueError:
            return "not valid JSON"
        if name == "vocab.json" and (not isinstance(parsed, dict) or len(parsed) < 40000):
            return "a vocabulary should have tens of thousands of entries"
        if name == "tokenizer.json" and (not isinstance(parsed, dict) or "model" not in parsed):
            return "not a tokenizer definition"
    elif name == "merges.txt" and text.count("\n") < 10000:
        return "too few merge rules"
    return None


class TokenizerFiles(ABC):
    @abstractmethod
    def present(self, key: str) -> tuple[str, ...]:
        """Names of the files of set ``key`` already in place."""

    @abstractmethod
    def download(self, key: str) -> tuple[str, ...]:
        """Fetch what is missing of set ``key``; returns the files written. Raises ``TokenizerDownloadError``."""



