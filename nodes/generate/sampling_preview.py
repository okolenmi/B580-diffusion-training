"""Sampling Preview: wired into a Sampler, it shows the picture forming on its own card while the sampler works."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

from engine.devices import Accelerator, part_dtype, resolve_device
from engine.generate.preview import SamplingPreview
from families.api.parts import VAEPart
from ..family_lookup import family_of
from graph import DisplayModule, Node, Port


class SamplingPreviewNode(Node):
    DISPLAY_NAME = "Sampling Preview"
    CATEGORY = "Generation"
    READS_SETTINGS = ("device.default", "precision.vae")
    INPUTS = {
        "vae": Port("vae", VAEPart, doc="The same VAE Decode Image uses. It is loaded when the first preview is made and dropped when sampling ends."),
        "every": Port("every", int, required=False, default=5, minimum=1, label="Every N steps",
                      doc="A preview costs a decode; the last step is always shown."),
        "size": Port("size", int, required=False, default=384, minimum=64, maximum=1024, step=64, unit="px",
                     doc="Longest side of the preview. Smaller needs less memory beside the model; if it still does not fit, previews stop for the run."),
    }
    OUTPUTS = {"preview": Port("preview", SamplingPreview, doc="Wire it into a Sampler's `preview` input.")}
    MODULES = (DisplayModule("preview", "image", "Preview", value_key="images", size=(6, 6),
                             doc="The newest look at the picture the sampler is making."),)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: VAEPart = inputs["vae"]
        family = family_of(self.context, part.family_id)
        device = resolve_device(self.context.settings.get("device.default"))
        dtype = part_dtype(part.dtype, self.context.settings.get("precision.vae"), torch.float32)
        held: list[Any] = []

        def decode(latents: torch.Tensor) -> torch.Tensor:
            if not held:
                with self.context.task("Loading the VAE for the preview"):
                    held.append(family.load_vae(part, device=device, dtype=dtype))
            with torch.no_grad():
                return held[0].decode(latents)

        def close() -> None:
            if held:
                held.pop().release()
                gc.collect()
                Accelerator(device).empty_cache()

        folder = self.context.env.service("artifacts") if self.context.env.has("artifacts") else None
        token = int(time.time())

        def show(pixels: torch.Tensor, step: int, total: int) -> None:
            path = folder.put_view(self.uid, f"{token}_{step:04d}", pixels)
            self.context.report(f"node:{self.uid}", {"step": step, "total": total, "images": [
                {"path": path, "width": pixels.shape[-1], "height": pixels.shape[-2]}]})

        def note(message: str) -> None:
            self.context.report("warning", {"node": self.uid, "message": message})

        preview = SamplingPreview(every=int(inputs.get("every") or 5), size=int(inputs.get("size") or 384),
                                  spatial_factor=family.latent_spec.spatial_factor, decode=decode, show=show, note=note, close=close)
        if folder is None:
            note("Sampling Preview: this run has no folder for pictures, nothing is shown")
            preview.stopped = True
        return {"preview": preview}
