"""Live preview: an addon of the Sampler that shows the picture forming on the nodes downstream that can show it (Preview Image)."""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

from engine.devices import Accelerator, part_dtype, resolve_device
from engine.generate.artifacts import ArtifactFolder
from engine.generate.preview import LatentRGB, LivePreview, PreviewDecoder, PreviewUnavailable, VAEPreview
from families.api.parts import VAEPart
from graph import Addon, Group, Port

METHODS = ("latent2rgb", "vae")


class LivePreviewAddon(Addon):
    """Every few steps, a look at the first picture of the batch: by default read straight off the latents (free, blurry), or decoded by the VAE (sharper, costs memory)."""

    DISPLAY_NAME = "Live preview"
    SLOT = "preview"
    FITS = ("SamplerNode",)
    GROUPS = (Group("decoder", "VAE decoder", visible_when=("method", "vae"), doc="Only for the vae method."),)
    INPUTS = {
        "every": Port("every", int, required=False, default=2, minimum=1, label="Every N steps", doc="The last step is always shown."),
        "method": Port("method", str, required=False, default="latent2rgb", choices=METHODS,
                       doc="latent2rgb: one linear map of the latents, no model and no memory, a blurry picture. vae: the real decoder on shrunk latents, "
                           "sharper, needs the VAE wired in and some memory beside the denoiser."),
        "vae": Port("vae", VAEPart, required=False, group="decoder", doc="The same VAE Decode Image uses. Loaded at the first preview, dropped when sampling ends."),
        "size": Port("size", int, required=False, default=384, minimum=64, maximum=1024, step=64, unit="px", group="decoder",
                     doc="Longest side of a decoded preview. Smaller needs less memory; if it still does not fit, previews stop for the run."),
    }

    def build(self, **inputs) -> LivePreview:
        context = self.host.context
        method, vae = str(inputs.get("method") or "latent2rgb"), inputs.get("vae")
        size, owner = int(inputs.get("size") or 384), self.host.address
        folder = context.env.service("artifacts") if context.env.has("artifacts") else None
        token = int(time.time() * 1000)

        def choose(family: Any) -> PreviewDecoder | None:
            if not context.live_consumers("image"):
                return None  # nothing downstream shows it, so nothing is computed
            if folder is None:
                raise PreviewUnavailable("this run has no folder for pictures, nothing is shown")
            if method == "vae":
                return self._vae(family, vae, size)
            if not family.latent_spec.rgb:
                raise PreviewUnavailable(f"{family.label} has no latent colour projection; choose the vae method")
            return LatentRGB(family.latent_spec.rgb, family.latent_spec.rgb_bias)

        def show(pixels: torch.Tensor, step: int, total: int) -> None:
            path = folder.put_live(owner, pixels)
            context.send_live("image", {"step": step, "total": total, "live": True, "images": [
                {"path": path, "v": f"{token}-{step}", "width": pixels.shape[-1], "height": pixels.shape[-2]}]})

        def note(message: str) -> None:
            context.report("warning", {"node": owner, "message": message})

        return LivePreview(every=int(inputs.get("every") or 2), choose=choose, show=show, note=note)

    def _vae(self, family: Any, part: VAEPart | None, size: int) -> PreviewDecoder:
        context = self.host.context
        if part is None:
            raise PreviewUnavailable("the vae method needs a VAE wired to this addon")
        if part.family_id != family.id:
            raise PreviewUnavailable("the VAE belongs to another model family than the denoiser")
        device = resolve_device(context.settings.get("device.default"))
        dtype = part_dtype(part.dtype, context.settings.get("precision.vae"), torch.float32)

        def load() -> Any:
            with context.task("Loading the VAE for the preview"):
                return family.load_vae(part, device=device, dtype=dtype)

        def release(codec: Any) -> None:
            codec.release()
            gc.collect()
            Accelerator(device).empty_cache()

        return VAEPreview(size=size, spatial_factor=family.latent_spec.spatial_factor, load=load, release=release)
