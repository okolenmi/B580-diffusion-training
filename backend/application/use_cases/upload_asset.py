"""UploadAsset -- stream raw bytes to a path inside a kind.

Thin by design: every filesystem rule (sandbox, extension allowlist,
overwrite refusal, size cap) belongs to the port's adapter, which is
where the disk is. What lives here is the one thing that is not the
adapter's business -- turning a client-supplied kind/path pair into a
validated request -- and the session the route drives.

**Why a session instead of one ``execute(body)``.** The only real caller
is an ``async def`` route reading a request stream, and the only thing
it may not do is block the loop (quality rule 5). So the reads have to
happen on the loop and the writes off it, which means *something* has to
interleave the two. Putting that in the route keeps the application
layer free of async plumbing while still giving it the whole decision:
`begin` performs every validation and opens the write, and the route
only moves bytes.

`execute` remains for callers that already hold a chunk iterable and do
not care where it came from (the tests). It streams that iterable; it
does not assemble it.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..ports.asset_store import AssetStore, AssetUploadWriter
from ..requests import AssetRequest


class UploadSession:
    """One validated upload in progress.

    A context manager, so the abort on an exception path is the
    caller's `with` and not a try/finally they have to remember.
    """

    def __init__(self, writer: AssetUploadWriter) -> None:
        self._writer = writer

    def write(self, chunk: bytes) -> None:
        """Append one chunk. Blocking -- call it off the event loop."""
        self._writer.write(chunk)

    def finish(self) -> str:
        """Commit and return the absolute path. Blocking."""
        return self._writer.finish()

    def abort(self) -> None:
        """Give up: no final file, no partial. Never raises."""
        self._writer.abort()

    def __enter__(self) -> "UploadSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.abort()  # a no-op after a successful finish()
        return False


class UploadAsset:
    def __init__(self, *, assets: AssetStore) -> None:
        self._assets = assets

    def begin(self, kind: str, relative_path: str, *,
              overwrite: bool = False) -> UploadSession:
        """Validate and open the write. Nothing is written until `write`.

        Every check runs before any byte is written, so a refused upload
        leaves no directory, no file and no partial (docs 07, quality
        rule 3).
        """
        request = AssetRequest.of(kind, relative_path, path_required=True)
        writer = self._assets.begin_upload(
            request.kind, request.relative_path, overwrite=overwrite
        )
        return UploadSession(writer)

    def execute(self, kind: str, relative_path: str,
                chunks: Iterable[bytes], *, overwrite: bool = False) -> str:
        """Stream `chunks` and return the final absolute path."""
        with self.begin(kind, relative_path, overwrite=overwrite) as session:
            for chunk in chunks:
                session.write(chunk)
            return session.finish()


__all__ = ["UploadAsset", "UploadSession"]
