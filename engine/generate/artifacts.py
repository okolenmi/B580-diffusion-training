"""The folder nodes leave pictures in for a page to show, one subfolder per node, the newest few kept."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .recipe import Image, ImageStore


class ArtifactFolder:
    """Pictures a running graph shows. A name holds the recipe and a digest of everything that made the picture (model and adapters
    included), so a picture never overwrites a different one and a page that caches by address cannot show an old picture."""

    KEEP = 48

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, owner: str, images: list[Image]) -> list[str]:
        """Writes the images under ``owner`` (a node uid); the paths come back relative to the folder, as the page asks for them."""
        folder = self.root / owner
        store = ImageStore(folder)
        paths = [f"{owner}/{store.save(image, self.stem(image)).name}" for image in images]
        self._prune(folder, keep=max(self.KEEP, len(images)))
        return paths

    @staticmethod
    def stem(image: Image) -> str:
        made = json.dumps([asdict(image.recipe), image.meta.get("model"), image.meta.get("adapters")], sort_keys=True, default=str)
        return f"{image.recipe.stem}_{hashlib.md5(made.encode()).hexdigest()[:8]}"

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
