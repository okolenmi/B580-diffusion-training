"""The vocabulary files the prompt tokenizers read, kept in ``<project>/assets/<set>_tokenizer``.

They are published data (OpenAI's CLIP vocabulary, Qwen's and T5's tokenizer files), not code, so they are downloaded once from the
model hubs instead of being found inside someone else's application. The trainer's loaders check this folder before anything else.
"""

from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
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


class HubTokenizerFiles(TokenizerFiles):
    def __init__(self, project_root: Path, timeout: float = 60.0) -> None:
        self._root = project_root
        self._timeout = timeout

    def present(self, key: str) -> tuple[str, ...]:
        folder = folder_of(self._root, key)
        return tuple(name for name in set_by_key(key).files if (folder / name).is_file())

    def download(self, key: str) -> tuple[str, ...]:
        wanted = set_by_key(key)
        folder = folder_of(self._root, key)
        written = []
        for name, urls in wanted.files.items():
            if (folder / name).is_file():
                continue
            data = self._fetch(key, name, urls)
            folder.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=folder, suffix=".part")
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            os.replace(temporary, folder / name)
            written.append(name)
        return tuple(written)

    def _fetch(self, key: str, name: str, urls: tuple[str, ...]) -> bytes:
        problems = []
        for url in urls:
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "trainer-setup"}), timeout=self._timeout) as reply:
                    data = reply.read()
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                problems.append(f"{url}: {exc}")
                continue
            bad = check_content(name, data)
            if bad:
                problems.append(f"{url}: {bad}")
                continue
            return data
        raise TokenizerDownloadError(f"could not download {name}. " + " | ".join(problems)
                                     + " Put the file into " + str(folder_of(self._root, key)) + " by hand if this "
                                     "machine cannot reach the hub.")

