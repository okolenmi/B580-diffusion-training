"""Auto Caption: write captions with a caption model for the images of a set that have none."""

from __future__ import annotations

import gc
from dataclasses import replace
from typing import Any

from ...data.captioning import CAPTIONERS, caption_images, make_captioner
from ...data.library import CHANNELS
from ...engine.devices import Accelerator, resolve_device
from ...graph import Group, Node, Port
from .handles import ImageSet


class AutoCaptionNode(Node):
    DISPLAY_NAME = "Auto Caption"
    CATEGORY = "Dataset"
    READS_SETTINGS = ("device.default",)
    GROUPS = (Group("model", "Model"), Group("text", "Text", collapsed=True))
    INPUTS = {
        "images": Port("images", ImageSet),
        "dataset": Port("dataset", str, picker="dataset", group="model",
                    doc="The dataset the images belong to; the captions are written into its library."),
        "channel": Port("channel", str, required=False, default="text", choices=CHANNELS, group="model",
                        doc="Caption channel to fill (the prompts node decides which channel training reads)."),
        "backend": Port("backend", str, required=False, default="transformers", group="model",
                        doc=f"Caption backend by name; registered: {', '.join(sorted(CAPTIONERS))}."),
        "model_id": Port("model_id", str, required=False, default="Salesforce/blip-image-captioning-base", group="model",
                         doc="A Hugging Face id or a local folder. The default writes short plain-English captions; "
                             "it downloads on first use, so the machine needs network access once."),
        "max_new_tokens": Port("max_new_tokens", int, required=False, default=40, minimum=1, group="model",
                               doc="Longest caption the model may write (CLIP reads 77 tokens)."),
        "overwrite": Port("overwrite", bool, required=False, default=False, group="model",
                          doc="Off: only images with an empty caption are captioned, so a rerun resumes and never replaces captions "
                              "you wrote. On: replace every caption."),
        "limit": Port("limit", int, required=False, default=0, minimum=0, group="model",
                      doc="Caption at most this many images in this run; 0 = all. A cheap trial before the full pass."),
        "prefix": Port("prefix", str, required=False, default="", group="text", doc="Trigger words before each caption, joined with ', '."),
        "suffix": Port("suffix", str, required=False, default="", group="text", doc="Words after each caption, joined with ', '."),
        "lowercase": Port("lowercase", bool, required=False, default=False, group="text",
                          doc="Lowercase the model's text (not the prefix and suffix)."),
    }
    OUTPUTS = {"images": Port("images", ImageSet, doc="The same images with the new captions in place; wire it on to the prompts."),
               "summary": Port("summary", str, doc="How many images were captioned, and why any were not.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        images: ImageSet = inputs["images"]
        channel = inputs.get("channel") or "text"
        device = resolve_device(self.context.settings.get("device.default"))
        library = self.context.env.service("datasets").library(inputs["dataset"])
        captioner = None
        try:
            with self.context.task("Loading the caption model"):
                captioner = make_captioner(inputs.get("backend") or "transformers", model_id=inputs.get("model_id"), device=str(device),
                                           max_new_tokens=inputs.get("max_new_tokens") or 40)
            with self.context.task("Captioning", total=len(images)) as task:
                report = caption_images(library, images, captioner, channel=channel, overwrite=bool(inputs.get("overwrite")),
                                        prefix=inputs.get("prefix") or "", suffix=inputs.get("suffix") or "",
                                        lowercase=bool(inputs.get("lowercase")), limit=inputs.get("limit") or None,
                                        should_stop=self.context.should_cancel, progress=lambda done, total: task.update(done, total))
        finally:
            if captioner is not None:
                captioner.close()
            library.close()
            gc.collect()
            Accelerator(device).empty_cache()
        self.context.report("captions", {"node": self.uid, "summary": report.summary(), "errors": report.errors})
        captions = {**images.captions, channel: {**images.captions.get(channel, {}), **report.written}}
        return {"images": replace(images, captions=captions), "summary": report.summary()}
