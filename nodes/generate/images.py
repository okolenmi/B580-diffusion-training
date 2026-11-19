"""Save Image: PNG files that carry the recipe of each image in their text chunks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from engine.generate.recipe import Image, ImageStore
from engine.handles import Images
from graph import Node, Port


def as_images(images: Images) -> list[Image]:
    """One ``Image`` per recipe, with what made it (the recipe, the model, the adapters) as its metadata."""
    shared = {k: v for k, v in images.meta.items() if k != "recipes"}
    return [Image(recipe, images.pixels[i], {"recipe": asdict(recipe), **shared}) for i, recipe in enumerate(images.recipes)]


class SaveImageNode(Node):
    DISPLAY_NAME = "Save Image"
    CATEGORY = "Output"
    INPUTS = {
        "images": Port("images", Images),
        "directory": Port("directory", str, required=False, default="output/images"),
    }
    OUTPUTS = {"saved": Port("saved", int, doc="Files written.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        store = ImageStore(inputs.get("directory") or "output/images")
        paths = [store.save(image) for image in as_images(inputs["images"])]
        self.context.report("saved_images", {"node": self.uid, "files": [str(p) for p in paths]})
        return {"saved": len(paths)}
