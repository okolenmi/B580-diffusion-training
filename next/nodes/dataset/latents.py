"""Latent Cache: encode the images with the graph's VAE, once per VAE."""

from __future__ import annotations

from typing import Any

from ...data.ids import vae_identity
from ...data.images import CROP_MODES, SizePolicy
from ...data.latent_cache import LatentCache
from ...families.api.header import read_header
from ...families.api.parts import VAEPart
from ...families.registry import get_family
from ...graph import Group, Node, Port
from .handles import ImageSet, LatentSet

_SERVICE = "datasets"


class LatentCacheNode(Node):
    DISPLAY_NAME = "Latent Cache"
    CATEGORY = "Dataset"
    GROUPS = (Group("size", "Size policy"), Group("encode", "Encoding", collapsed=True))
    INPUTS = {
        "images": Port("images", ImageSet),
        "vae": Port("vae", VAEPart),
        "dataset": Port("dataset", str, picker="dataset", doc="The dataset the images came from; its folder holds the caches."),
        "base": Port("base", int, required=False, default=1024, minimum=64, step=64, unit="px", group="size",
                     doc="Short side of every image after resizing."),
        "mode": Port("mode", str, required=False, default="fit", choices=CROP_MODES, group="size",
                     doc="fit keeps aspect ratio (long images become several crops); the others make squares."),
        "max_aspect": Port("max_aspect", float, required=False, default=2.0, minimum=1.0, group="size",
                           visible_when=("mode", "fit"), doc="Longest allowed side as a multiple of the base."),
        "batch_size": Port("batch_size", int, required=False, default=4, minimum=1, group="encode",
                           doc="Images encoded at once."),
        "device": Port("device", str, required=False, default="auto", choices=("auto", "cpu", "xpu"), group="encode"),
    }
    OUTPUTS = {"latents": Port("latents", LatentSet, doc="Latents for the images under this size policy.")}

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: VAEPart = inputs["vae"]
        family = get_family(part.family_id)
        spec = family.latent_spec
        policy = SizePolicy(base=inputs.get("base") or 1024, mode=inputs.get("mode") or "fit",
                            multiple=int(spec.spatial_factor), max_aspect=inputs.get("max_aspect") or 2.0)
        images: ImageSet = inputs["images"]
        header = read_header(part.source.path)
        identity = vae_identity(header, part.source.prefix, spec)
        cache = LatentCache(self.context.env.service(_SERVICE).caches(inputs["dataset"]), identity, spec,
                            label=f"{part.family_id} VAE {part.source.path.rsplit('/', 1)[-1]}")
        todo = cache.missing(images, policy)
        skipped = False
        if todo:
            codec = family.load_vae(part, device=_device(inputs.get("device") or "auto"))
            try:
                stats = cache.build(images, policy, codec, batch_size=inputs.get("batch_size") or 4,
                                    progress=lambda done, total: self.context.report(
                                        "cache_progress", {"node": self.uid, "done": done, "total": total}),
                                    should_stop=self.context.should_cancel)
            finally:
                codec.release()
            if stats.stopped:
                raise RuntimeError("Latent Cache: stopped before every image was encoded; run again to continue")
            if stats.skipped:
                skipped = True
                self.context.report("warning", {"node": self.uid, "skipped": stats.skipped[:20]})
        return {"latents": cache.view(images, policy, skip_missing=skipped)}


def _device(choice: str) -> str:
    if choice != "auto":
        return choice
    import torch

    return "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
