"""Save Image: PNG files that carry the recipe of each image in their text chunks."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from engine.generate.recipe import Image, ImageStore
from engine.handles import Images
from graph import DisplayModule, Node, Port


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


class PreviewImageNode(Node):
    DISPLAY_NAME = "Preview Image"
    CATEGORY = "Output"
    INPUTS = {"images": Port("images", Images)}
    OUTPUTS = {"images": Port("images", Images, doc="The images, unchanged, so a preview can sit in front of Save Image.")}
    LIVE_FRAMES = ("image",)  # a Sampler upstream shows the picture forming here (its Live preview addon); the finished images then replace it
    MODULES = (DisplayModule("images", "image", "Images", value_key="images", size=(6, 6),
                             doc="The picture while it forms, then the images of the latest run, newest first."),)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        if not self.context.env.has("artifacts"):
            self.context.report("warning", {"node": self.uid, "message": "Preview Image: this run has no folder for pictures, nothing is shown"})
            return {"images": inputs["images"]}
        images = as_images(inputs["images"])
        paths = self.context.env.service("artifacts").put(self.uid, images)
        shown = [{"path": path, "seed": i.recipe.seed, "width": i.pixels.shape[-1], "height": i.pixels.shape[-2]} for path, i in zip(paths, images)]
        self.context.report(f"node:{self.uid}", {"step": 1, "images": shown})
        return {"images": inputs["images"]}
