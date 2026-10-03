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

``resolve`` deliberately keeps an absolute path as sent, because not every
caller wants the project root -- the asset store serves from the ComfyUI
model directories, which live elsewhere. That is only safe while every
caller adds its own containment, and ``config`` did not: the raw config
editor took any path the client named, so

    GET /api/v1/config/raw?path=../../../../etc/passwd

returned the file, and the same value on ``PUT`` wrote it. Both were 200.
``config`` therefore carries its own containment, which is what "edit this
project's configuration" means anyway.
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
        """Absolute stays as sent; relative hangs off the project root.

        No containment here -- see the module docstring. Use
        :meth:`config` for a path that must stay inside the project.
        """
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else self.root / candidate

    def config(self, raw: str) -> Path:
        """A config path inside this project, or 422 naming the field.

        The common case, spelled out at the call site so the intent reads
        as ``paths.config(...)``.

        Confined to the project root, which is the whole contract: these
        four routes are an editor for *this project's* configuration, and
        a client-named path that walks out of the project is not a config
        path. Both halves are checked against the resolved root rather than
        the textual one, so a symlink inside the tree pointing out of it is
        refused too.

        Absolute paths are still allowed, but only when they land inside
        the root -- callers legitimately hold an absolute path to a config
        they are about to write, and that should keep working.
        """
        if not raw:
            raise InvalidQueryError("config path is required")
        if "\x00" in raw:
            # Not a path: it is where the string ends. Every filesystem
            # call below raises ValueError on one, which would reach the
            # client as a 500.
            raise InvalidQueryError("config path must not contain a NUL byte")

        candidate = self.resolve(raw)
        try:
            resolved = candidate.resolve()
            root = self.root.resolve()
        except OSError as exc:  # e.g. a loop of symlinks
            raise InvalidQueryError(f"config path cannot be resolved: {exc}") from exc

        if not resolved.is_relative_to(root):
            raise InvalidQueryError(
                f"config path must be inside the project: {raw!r} resolves "
                f"outside {root}"
            )
        return candidate