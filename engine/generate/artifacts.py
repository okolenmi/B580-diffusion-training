"""The folder nodes leave pictures in for a page to show, one subfolder per node, the newest few kept."""

from __future__ import annotations

from pathlib import Path

from .recipe import Image, ImageStore


class ArtifactFolder:
    """Pictures a running graph shows. A name follows the recipe, so the same recipe overwrites its own file."""

    KEEP = 48

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, owner: str, images: list[Image]) -> list[str]:
        """Writes the images under ``owner`` (a node uid); the paths come back relative to the folder, as the page asks for them."""
        folder = self.root / owner
        store = ImageStore(folder)
        paths = [f"{owner}/{store.save(image).name}" for image in images]
        self._prune(folder, keep=max(self.KEEP, len(images)))
        return paths

    @staticmethod
    def _prune(folder: Path, keep: int) -> None:
        files = sorted(folder.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            stale.unlink(missing_ok=True)
