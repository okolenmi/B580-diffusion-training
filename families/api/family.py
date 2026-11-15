"""What a family module must provide. Grows with each phase; phase 2 needs only the fingerprint."""

from __future__ import annotations

from typing import Protocol

from .detection import Match
from .header import HeaderIndex


class Fingerprint(Protocol):
    """``families/<id>/fingerprint.py``: pure, header-only, no torch."""

    FAMILY_ID: str
    LABEL: str

    def match(self, index: HeaderIndex) -> Match | None: ...
