"""Tokenizer files fetched from the Hugging Face hub."""

from __future__ import annotations

import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from ..application.ports.tokenizer_files import TokenizerDownloadError, set_by_key, folder_of, check_content, TokenizerFiles


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
