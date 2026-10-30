"""Is this project's renderer the same as ComfyUI's? The same recipes through both, image against image."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image

from ..accept.visual import ImageSheet
from .metrics import Change, Detail
from .render import Recipe
from .report import read_pixels


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    error = (a.float() - b.float()).pow(2).mean()
    return float("inf") if error == 0 else float(10 * torch.log10(255.0 ** 2 / error))


class EngineComparison:
    """Reads ours from ``ours`` (PNGs named by recipe stem) and renders the recipes in ComfyUI into ``theirs``."""

    def __init__(self, recipes: list[Recipe], ours: Path, theirs: Path, comfy: str, python: str, checkpoint: str, lora: str | None, dtype: str, device: str):
        self.recipes, self.ours, self.theirs = recipes, ours, theirs
        self.comfy, self.python, self.checkpoint, self.lora, self.dtype, self.device = comfy, python, checkpoint, lora, dtype, device

    def render_theirs(self) -> None:
        self.theirs.mkdir(parents=True, exist_ok=True)
        spec = self.theirs / "recipes.json"
        spec.write_text(json.dumps([{**asdict(r), "stem": r.stem} for r in self.recipes]), encoding="utf8")
        command = [self.python, str(Path(__file__).resolve().parents[1] / "accept" / "comfy_render.py"), "--comfy", self.comfy, "--checkpoint", self.checkpoint,
                   "--recipes", str(spec), "--out", str(self.theirs), "--unet-dtype", self.dtype, "--device", self.device, *(["--lora", self.lora] if self.lora else [])]
        print("running ComfyUI:", " ".join(command), flush=True)
        if subprocess.run(command).returncode:
            raise SystemExit("the ComfyUI render failed (see its output above)")

    def markdown(self) -> str:
        rows = ["| image | PSNR | mean abs difference | coarse change | fine change | finest octave: ComfyUI / ours | spectral slope ours / ComfyUI |", "|---|---|---|---|---|---|---|"]
        sheet = ImageSheet((384, 384), columns=2, title="left: this project, right: ComfyUI")
        psnrs = []
        for recipe in self.recipes:
            mine, theirs = self.ours / f"{recipe.stem}.png", self.theirs / f"{recipe.stem}.png"
            if not (mine.exists() and theirs.exists()):
                continue
            a, b = read_pixels(mine), read_pixels(theirs)
            change = Change.between(a, b)
            psnrs.append(psnr(a, b))
            da, db = Detail.of(a), Detail.of(b)
            rows.append(f"| {(recipe.prompt or '(empty)')[:30]} seed {recipe.seed} x{recipe.strength:g} | {psnrs[-1]:.1f} dB | {float((a.float() - b.float()).abs().mean()):.2f} | {change.coarse:.4f} | "
                        f"{change.fine:.4f} | {1 / max(change.top_band_ratio, 1e-9):.2f} | {da.slope:.2f} / {db.slope:.2f} |")
            sheet.add(Image.open(mine).convert("RGB"), f"ours seed {recipe.seed}")
            sheet.add(Image.open(theirs).convert("RGB"), "ComfyUI")
        sheet.save(self.ours.parent / "engines-sheet.png")
        head = ["# Renderer against ComfyUI", "", f"{len(psnrs)} images, mean PSNR {sum(psnrs) / max(len(psnrs), 1):.1f} dB (above ~35 dB the images are hard to tell apart; ComfyUI runs the UNet in "
                f"{self.dtype if self.dtype != 'auto' else 'its own choice of precision'}, this project in bf16).", ""]
        return "\n".join(head + rows)
