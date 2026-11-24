"""The folder nodes leave pictures in for a page to show, one subfolder per node, the newest few kept."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

    def put_view(self, owner: str, name: str, pixels: Any) -> str:
        """One plain picture (uint8 ``(3, H, W)``) under ``owner`` named ``name``; what a live preview writes at every look."""
        from PIL import Image as PILImage

        folder = self.root / owner
        folder.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(pixels.permute(1, 2, 0).numpy()).save(folder / f"{name}.png")
        self._prune(folder, keep=self.KEEP)
        return f"{owner}/{name}.png"

    @staticmethod
    def _prune(folder: Path, keep: int) -> None:
        files = sorted(folder.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            stale.unlink(missing_ok=True)
