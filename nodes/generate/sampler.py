"""Sampler: noise to latents with a denoiser and an encoded prompt, one image per seed."""

from __future__ import annotations

import gc
from dataclasses import asdict
from typing import Any

import torch

from engine.devices import Accelerator, resolve_device, resolve_dtype
from engine.generate.recipe import Recipe
from engine.generate.sampling import SAMPLERS, GuidedDenoiser, NoiseTable, SigmaSchedule, initial_noise
from engine.handles import Latents, PromptEncoding
from families.api.parts import DenoiserPart
from families.registry import get_family
from graph import DisplayModule, Group, Node, Port
from nodes.model.merges import apply_merges


class SamplerNode(Node):
    DISPLAY_NAME = "Sampler"
    CATEGORY = "Generation"
    READS_SETTINGS = ("device.default", "precision.denoiser")
    GROUPS = (Group("image", "Image"), Group("sampling", "Sampling"))
    INPUTS = {
        "denoiser": Port("denoiser", DenoiserPart, doc="From Load Model, optionally through Adapter Stack or Merge Adapter."),
        "encoding": Port("encoding", PromptEncoding),
        "seed": Port("seed", int, required=False, default=0, minimum=0,
                     doc="The first image's seed; image i of a batch uses seed + i, so every image can be made again alone.", group="sampling"),
        "steps": Port("steps", int, required=False, default=25, minimum=1, maximum=200, group="sampling"),
        "cfg": Port("cfg", float, required=False, default=6.0, minimum=1.0, step=0.5, label="Guidance",
                    doc="How hard the image follows the prompt over the negative one. 1 skips the negative pass.", group="sampling"),
        "sampler": Port("sampler", str, required=False, default="dpmpp_2m", choices=tuple(SAMPLERS), group="sampling"),
        "scheduler": Port("scheduler", str, required=False, default="normal", choices=tuple(SigmaSchedule.NAMES), group="sampling"),
        "width": Port("width", int, required=False, default=1024, minimum=64, maximum=2048, step=8, unit="px", group="image"),
        "height": Port("height", int, required=False, default=1024, minimum=64, maximum=2048, step=8, unit="px", group="image"),
        "batch": Port("batch", int, required=False, default=1, minimum=1, maximum=16, group="image"),
        "eps_scale": Port("eps_scale", float, required=False, default=1.0, minimum=1.0, step=0.01, advanced=True,
                          doc="Epsilon scaling: the guided prediction is divided by this, since sampling errors make it a little too large.", group="sampling"),
    }
    OUTPUTS = {"latents": Port("latents", Latents)}
    MODULES = (DisplayModule("progress", "progress", "Sampling", value_key="step", total_key="total", size=(12, 2),
                             doc="Denoising steps done, of the steps asked for."),)

    def build(self, **inputs) -> dict[str, Any]:
        self.validate_inputs(inputs)
        part: DenoiserPart = inputs["denoiser"]
        encoding: PromptEncoding = inputs["encoding"]
        if encoding.family is not get_family(part.family_id):
            raise ValueError("Sampler: the prompt was encoded for a different model family than the denoiser")
        width, height = int(inputs.get("width") or 1024), int(inputs.get("height") or 1024)
        if width % 8 or height % 8:
            raise ValueError(f"Sampler: the size {width}x{height} must be a multiple of 8")
        steps, batch = int(inputs.get("steps") or 25), int(inputs.get("batch") or 1)
        recipes = tuple(Recipe(prompt=encoding.prompt, negative=encoding.negative_prompt, seed=int(inputs.get("seed") or 0) + i, steps=steps,
                               cfg=float(inputs.get("cfg") or 6.0), sampler=inputs.get("sampler") or "dpmpp_2m",
                               scheduler=inputs.get("scheduler") or "normal", width=width, height=height,
                               eps_scale=float(inputs.get("eps_scale") or 1.0)) for i in range(batch))
        first = recipes[0]
        family = get_family(part.family_id)
        device = resolve_device(self.context.settings.get("device.default"))
        precision = part.dtype if part.dtype != "auto" else self.context.settings.get("precision.denoiser")
        with self.context.task("Loading the denoiser"):
            denoiser = family.load_denoiser(part, device=device, dtype=resolve_dtype(precision, torch.bfloat16))
        try:
            apply_merges(family, denoiser, part.merges, lambda: self.context.env.service("models"), self.context.task)
            denoiser.eval()
            latents = self._sample(denoiser, encoding, recipes, first, device)
        finally:
            del denoiser
            gc.collect()
            Accelerator(device).empty_cache()
        adapters = [{"file": m.file, "strength": m.strength} for m in part.merges]
        return {"latents": Latents(latents, recipes, {"model": part.source.path.rsplit("/", 1)[-1], "adapters": adapters,
                                                      "recipes": [asdict(r) for r in recipes]})}

    def _sample(self, denoiser: Any, encoding: PromptEncoding, recipes: tuple[Recipe, ...], first: Recipe, device: str) -> torch.Tensor:
        table = NoiseTable()
        size, batch = (first.height, first.width), len(recipes)
        positive = encoding.condition(size, batch).to(device)
        negative = encoding.condition(size, batch, negative=True).to(device)
        total, done = first.steps, 0

        def network(x_in: torch.Tensor, t: torch.Tensor, is_positive: bool) -> torch.Tensor:
            nonlocal done
            if self.context.should_cancel():
                raise InterruptedError("stopped")
            if is_positive:
                self.context.report(f"node:{self.uid}", {"step": done, "total": total})
                done += 1
            return denoiser.predict(x_in.to(device), t.to(device), positive if is_positive else negative).float()

        sigmas = SigmaSchedule(first.scheduler, table).sigmas(first.steps)
        noise = torch.cat([initial_noise(r.seed, (1, 4, r.height // 8, r.width // 8), float(sigmas[0]), table) for r in recipes]).to(device)
        cfg = 1.0 if first.prompt == first.negative else first.cfg  # equal prompts predict equal noise: the second pass would change nothing
        with torch.no_grad(), self.context.task("Sampling", total=total):
            latents = SAMPLERS[first.sampler]().run(GuidedDenoiser(network, table, cfg, first.eps_scale), noise, sigmas).cpu()
        self.context.report(f"node:{self.uid}", {"step": total, "total": total})
        return latents
