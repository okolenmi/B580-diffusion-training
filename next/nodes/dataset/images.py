"""Dataset Library: pick a dataset and a set in it."""

from __future__ import annotations

from typing import Any

from ...data.library import CHANNELS, DEFAULT_SET
from ...graph import Group, Node, Port
from .handles import ImageSet

_SERVICE = "datasets"


class DatasetLibraryNode(Node):
    DISPLAY_NAME = "Dataset Library"
    CATEGORY = "Dataset"
    GROUPS = (Group("filter", "Filter", collapsed=True),)
    INPUTS = {
        "dataset": Port("dataset", str, picker="dataset", doc="A dataset created on the Datasets page."),
        "set": Port("set", str, required=False, default=DEFAULT_SET, doc="Training set inside the dataset."),
        "min_side": Port("min_side", int, required=False, default=0, minimum=0, unit="px", group="filter",
                         doc="Skip images whose shorter side is smaller than this."),
        "caption_required": Port("caption_required", str, required=False, default="", group="filter",
                                 choices=("", *CHANNELS), doc="Skip images with no caption in this channel."),
    }
    OUTPUTS = {"images": Port("images", ImageSet, doc="Image ids, paths, sizes and captions. No pixels are read.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        library = self.context.env.service(_SERVICE).library(inputs["dataset"])
        try:
            images = library.snapshot(inputs.get("set") or DEFAULT_SET, min_side=inputs.get("min_side") or 0,
                                      caption_required=inputs.get("caption_required") or None)
        finally:
            library.close()
        if not len(images):
            raise ValueError(f"Dataset Library: {inputs['dataset']!r} has no images in set "
                             f"{inputs.get('set') or DEFAULT_SET!r} after the filters")
        return {"images": images}
