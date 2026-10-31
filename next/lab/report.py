"""The comparison of an experiment's arms: tables of measurements, contact sheets, and a folder small enough to send."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..accept.visual import ImageSheet
from .experiment import Experiment
from .metrics import Change, Detail, mean
from .render import Recipe

DETAIL_FIELDS = ("slope", "fine_share", "sharpness")
CHANGE_FIELDS = ("coarse", "fine", "fine_to_coarse", "top_band_ratio", "sharpness_ratio")


def read_pixels(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).clone()


class ImageMeasurements:
    """Detail of every image and its change against the reference of the same seed, kept in ``image_metrics.json`` next to the images."""

    def __init__(self, experiment: Experiment):
        self.experiment, self.root = experiment, experiment.root

    def _entries(self, folder: Path) -> list[dict]:
        index = folder / "images" / "index.json"
        return json.loads(index.read_text()) if index.exists() else []

    def of(self, name: str) -> list[dict]:
        folder = self.root / ("reference" if name == "reference" else name)
        cache = folder / "image_metrics.json"
        entries = self._entries(folder)
        if cache.exists() and len(json.loads(cache.read_text())) == len(entries):
            return json.loads(cache.read_text())
        rows = []
        for entry in entries:
            recipe = Recipe(**entry["recipe"])
            pixels = read_pixels(folder / "images" / entry["file"])
            row = {"file": entry["file"], "prompt": recipe.prompt, "seed": recipe.seed, "strength": recipe.strength, **Detail.of(pixels).__dict__}
            row.pop("bands")
            reference = self.root / "reference" / "images" / f"{replace(recipe, strength=0.0).stem}.png"
            if name != "reference" and reference.exists():
                row.update(Change.between(pixels, read_pixels(reference)).__dict__)
            rows.append(row)
        cache.write_text(json.dumps(rows), encoding="utf8")
        return rows

    def by_strength(self, name: str) -> dict[float, dict[str, float]]:
        rows = self.of(name)
        out: dict[float, dict[str, float]] = {}
        for strength in sorted({r["strength"] for r in rows}):
            chosen = [r for r in rows if r["strength"] == strength]
            out[strength] = {k: mean([r[k] for r in chosen if k in r]) for k in (*DETAIL_FIELDS, *CHANGE_FIELDS)}
        return out


class Report:
    def __init__(self, experiment: Experiment):
        self.experiment, self.root = experiment, experiment.root
        self.measure = ImageMeasurements(experiment)
        self.trained = [a for a in experiment.arms if (self.root / a.name / "images" / "index.json").exists()]

    def _json(self, arm: str, name: str) -> dict:
        path = self.root / arm / name
        return json.loads(path.read_text()) if path.exists() else {}

    def arms_table(self) -> list[str]:
        rows = ["| arm | arguments | steps | minutes | mean loss, last 100 steps | low / mid / high third |", "|---|---|---|---|---|---|"]
        for arm in self.experiment.arms:
            train = self._json(arm.name, "train.json")
            recent = train.get("recent") or {}
            rows.append(f"| {arm.name} | `{' '.join(arm.args) or '-'}` | {train.get('steps', '-')} | {train.get('seconds', 0) / 60:.1f} | {recent.get('loss', float('nan')):.4f} | "
                        f"{recent.get('low', float('nan')):.4f} / {recent.get('mid', float('nan')):.4f} / {recent.get('high', float('nan')):.4f} |" if train
                        else f"| {arm.name} | `{' '.join(arm.args) or '-'}` | - | - | - | - |")
        return rows

    def detail_table(self, strength: float) -> list[str]:
        reference = self.measure.by_strength("reference").get(0.0) if (self.root / "reference").exists() else None
        head = ["| arm | spectral slope | fine-power share | sharpness | finest octave vs reference | sharpness vs reference | coarse change | fine change | fine / coarse |",
                "|---|---|---|---|---|---|---|---|---|"]
        rows = []
        if reference:
            rows.append(f"| reference (no adapter) | {reference['slope']:.2f} | {reference['fine_share']:.4f} | {reference['sharpness']:.5f} | 1 | 1 | - | - | - |")
        for arm in self.trained:
            m = self.measure.by_strength(arm.name).get(strength)
            if m:
                rows.append(f"| {arm.name} | {m['slope']:.2f} | {m['fine_share']:.4f} | {m['sharpness']:.5f} | {m['top_band_ratio']:.2f} | {m['sharpness_ratio']:.2f} | "
                            f"{m['coarse']:.4f} | {m['fine']:.4f} | {m['fine_to_coarse']:.2f} |")
        return head + rows

    def loss_table(self) -> list[str]:
        tables = {arm.name: self._json(arm.name, "timestep_loss.json") for arm in self.trained}
        if not any(tables.values()):
            return []
        splits = [s for s in ("train", "holdout") if any(s in t for t in tables.values())]
        steps = sorted({int(t) for tab in tables.values() for s in splits for t in tab.get(s, {})})
        head = "| t | " + " | ".join(f"{arm} {s}" for arm in tables for s in splits if s in tables[arm]) + " |"
        rows = [head, "|" + "---|" * (head.count("|") - 1)]
        for t in steps:
            cells = []
            for tab in tables.values():
                for s in splits:
                    if s in tab:
                        base, lora = tab[s][str(t)]
                        cells.append(f"{lora / base:.3f}")
            rows.append(f"| {t} | " + " | ".join(cells) + " |")
        return rows

    def probe_table(self) -> list[str]:
        rows = ["| arm | step | adapter/base loss low / mid / high third | worst |", "|---|---|---|---|"]
        found = False
        for arm in self.experiment.arms:
            probes = self._json(arm.name, "train.json").get("probes") or []
            if probes:
                found = True
                last = probes[-1]
                rows.append(f"| {arm.name} | {last['step']} | {last['low']:.3f} / {last['mid']:.3f} / {last['high']:.3f} | {last['worst']:.3f} |")
        return rows if found else []

    def sheets(self) -> list[Path]:
        """One overview per prompt, seed and strength (reference first, then the arms) and one native-size crop sheet per prompt and seed."""
        exp, out = self.experiment, self.root / "sheets"
        names = (["reference"] if (self.root / "reference").exists() else []) + [a.name for a in self.trained]
        made = []
        for prompt in exp.grid.prompts:
            for seed in exp.grid.seeds:
                crops = ImageSheet((320, 320), columns=len(names), title=f"{prompt[:60] or '(empty prompt)'} - seed {seed} - strength 1 - native pixels")
                for strength in exp.grid.strengths:
                    sheet = ImageSheet((384, 384), columns=min(4, len(names)), title=f"{prompt[:60] or '(empty prompt)'} - seed {seed} - strength {strength:g}")
                    for name in names:
                        recipe = Recipe(prompt=prompt, seed=seed, strength=0.0 if name == "reference" else strength, **{k: getattr(exp.grid, k) for k in ("negative", "steps", "cfg", "sampler", "scheduler", "width", "height")})
                        path = self.root / name / "images" / f"{recipe.stem}.png"
                        if path.exists():
                            sheet.add(Image.open(path).convert("RGB"), name)
                            if strength == exp.grid.strengths[-1]:
                                image = Image.open(path).convert("RGB")
                                left, top = (image.width - 320) // 2, (image.height - 320) // 2
                                crops.add(image.crop((left, top, left + 320, top + 320)), name)
                    if sheet.items:
                        made.append(sheet.save(out / f"{Recipe(prompt, seed).stem.rsplit('_x', 1)[0]}_x{strength:g}.png"))
                if crops.items:
                    made.append(crops.save(out / f"{Recipe(prompt, seed).stem.rsplit('_x', 1)[0]}_crops.png"))
        return made

    def resolution_notes(self) -> list[str]:
        """A LoRA trained at one resolution and rendered at another is a different experiment; say so."""
        args = [*self.experiment.train_args, *[a for arm in self.experiment.arms for a in arm.args]]
        trained = [int(args[i + 1]) for i, a in enumerate(args[:-1]) if a == "--resolution" and args[i + 1].isdigit()]
        grid = self.experiment.grid
        if trained and max(trained) < min(grid.width, grid.height):
            return [f"**Note:** training used `--resolution {max(trained)}`, the pictures are rendered at {grid.width}x{grid.height}. "
                    "Effects seen only at the larger size may come from that mismatch, not from the arm.", ""]
        return []

    def markdown(self, sheets: list[Path]) -> str:
        exp = self.experiment
        lines = [f"# Lab report: {exp.name}", "", exp.note, "", *self.resolution_notes(), "## Arms", "", *self.arms_table(), ""]
        for strength in exp.grid.strengths:
            lines += [f"## Detail of the images at strength {strength:g}", "",
                      "Ratios against the same prompt and seed rendered without the adapter: below 1 means less fine detail. Slope: more negative = less fine detail "
                      "(natural photographs about -2). Coarse change: difference of the picture reduced 16 times; fine change: difference of what a blur leaves out.", "",
                      *self.detail_table(strength), ""]
        loss = self.loss_table()
        if loss:
            lines += ["## Loss by timestep: adapter loss / base loss", "", "Below 1: better than the base model at that timestep; above 1: worse. `train` images were in the training set; `holdout` images were not.", "", *loss, ""]
        probe = self.probe_table()
        if probe:
            lines += ["## Probe at the end of training (training images)", "", *probe, ""]
        lines += ["## Pictures", "", *[f"- `{p.relative_to(self.root).as_posix()}`" for p in sheets], ""]
        return "\n".join(lines)

    def write(self) -> Path:
        sheets = self.sheets()
        path = self.root / "report.md"
        path.write_text(self.markdown(sheets), encoding="utf8")
        return path

    def pack(self, limit_mb: float = 25.0) -> Path:
        """``<experiment>-share.zip``: the report, every measurement and log, and the sheets (as JPEG, quality lowered until the zip fits)."""
        target = self.root / f"{self.experiment.name}-share.zip"
        for quality in (88, 75, 60, 45):
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
                for name in ("report.md", "experiment.json"):
                    if (self.root / name).exists():
                        archive.write(self.root / name, name)
                for arm in self.experiment.arms:
                    for name in ("train.json", "train.log", "timestep_loss.json", "image_metrics.json"):
                        if (self.root / arm.name / name).exists():
                            archive.write(self.root / arm.name / name, f"{arm.name}/{name}")
                    if (self.root / arm.name / "images" / "index.json").exists():
                        archive.write(self.root / arm.name / "images" / "index.json", f"{arm.name}/index.json")
                for sheet in sorted((self.root / "sheets").glob("*.png")):
                    jpeg = sheet.with_suffix(".jpg")
                    Image.open(sheet).convert("RGB").save(jpeg, quality=quality)
                    archive.write(jpeg, f"sheets/{jpeg.name}")
                    jpeg.unlink()
            if target.stat().st_size / 1e6 <= limit_mb:
                break
        return target
