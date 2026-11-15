"""The part of a latent cache the server can read without torch: where its index lives and the name stored in it."""

from __future__ import annotations

import sqlite3
from pathlib import Path

INDEX = "index.sqlite"


def cache_label(folder: str | Path) -> str:
    """The readable name stored with one cache, or "" when it has none or its index cannot be read (a damaged cache must
    stay listable so it can be deleted)."""
    index = Path(folder) / INDEX
    if not index.is_file():
        return ""
    try:
        db = sqlite3.connect(index)
        try:
            return dict(db.execute("select key, value from meta")).get("label", "")
        finally:
            db.close()
    except sqlite3.Error:
        return ""
