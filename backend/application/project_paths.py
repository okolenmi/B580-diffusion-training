"""ProjectPaths -- "a path the client named" as a value object.

Six use cases carried a byte-identical resolver::

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self._root / candidate

and five of them repeated the "path is required" guard with the same
wording. The rule is small; copying it six times is how one copy ends
up disagreeing with the others about what "relative" means. It lives
here instead, and the use cases take this object rather than a bare
``project_root: Path`` (docs 08 S-06).

``require`` and ``resolve`` are deliberately separate: a request that
omits a required argument must say so (422 ``invalid_query``) before any
path arithmetic happens, and the error names the field, so the message
does not have to be written out five times either.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidQueryError


@dataclass(frozen=True, slots=True)
class ProjectPaths:
    """Resolves client-supplied paths against the project root."""

    root: Path

    def require(self, raw: str, field: str = "path") -> Path:
        """The named path, or 422 naming the missing field."""
        if not raw:
            raise InvalidQueryError(f"{field} is required")
        return self.resolve(raw)

    def resolve(self, raw: str) -> Path:
        """Absolute stays as sent; relative hangs off the project root."""
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.root / candidate

    def config(self, raw: str) -> Path:
        """A config path -- the common case, spelled out at the call site
        so the intent reads as ``paths.config(...)``."""
        return self.require(raw, "config path")