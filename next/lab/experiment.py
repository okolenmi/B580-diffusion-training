"""An experiment: one training setup, the variants (arms) to compare, and the images to render and measure for each."""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

from .render import Grid


@dataclass(frozen=True)
class Arm:
    """One variant: a name and the extra training arguments that make it differ from the others."""

    name: str
    args: tuple[str, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class Experiment:
    name: str
    models: str
    datasets: str
    dataset: str
    checkpoint: str
    output: str
    arms: tuple[Arm, ...]
    grid: Grid
    train_args: tuple[str, ...] = ()
    device: str = "xpu"
    train_images: str = ""  # a folder of training images, for the per-timestep loss
    holdout_images: str = ""  # a folder of images the training never saw
    loss_images: int = 6
    loss_size: int = 768
    timesteps: tuple[int, ...] = ()
    reference: bool = True
    note: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.output).resolve() / self.name

    def arm(self, name: str) -> Arm:
        return next(a for a in self.arms if a.name == name)

    @staticmethod
    def expand(arms: list[dict], product: list[list[dict]]) -> tuple[Arm, ...]:
        """The listed arms, then every combination of one choice from each group of ``product`` (names joined with ``+``, arguments in order)."""
        out = [Arm(a["name"], tuple(a.get("args", ())), a.get("note", "")) for a in arms]
        if product:
            for combo in itertools.product(*product):
                out.append(Arm("+".join(c["name"] for c in combo), tuple(x for c in combo for x in c.get("args", ())), " ".join(c.get("note", "") for c in combo).strip()))
        names = [a.name for a in out]
        if len(set(names)) != len(names) or not names:
            raise ValueError("an experiment needs at least one arm and every arm needs its own name")
        return tuple(out)

    @staticmethod
    def from_dict(spec: dict) -> Experiment:
        grid = dict(spec.get("grid", {}))
        grid["prompts"] = tuple(grid.get("prompts") or ("",))
        for key in ("seeds", "strengths"):
            if key in grid:
                grid[key] = tuple(grid[key])
        images = spec.get("images", {})
        known = {"name", "models", "datasets", "dataset", "checkpoint", "output", "arms", "product", "grid", "train_args", "device", "images", "timesteps", "reference", "note"}
        return Experiment(name=spec["name"], models=spec["models"], datasets=spec["datasets"], dataset=spec["dataset"], checkpoint=spec["checkpoint"],
                          output=spec.get("output", "lab-out"), arms=Experiment.expand(spec.get("arms", []), spec.get("product", [])), grid=Grid(**grid),
                          train_args=tuple(spec.get("train_args", ())), device=spec.get("device", "xpu"), train_images=images.get("train", ""), holdout_images=images.get("holdout", ""),
                          loss_images=images.get("count", 6), loss_size=images.get("size", 768), timesteps=tuple(spec.get("timesteps", ())),
                          reference=spec.get("reference", True), note=spec.get("note", ""), extra={k: v for k, v in spec.items() if k not in known})

    @staticmethod
    def load(path: str | Path) -> Experiment:
        return Experiment.from_dict(json.loads(Path(path).read_text(encoding="utf8")))

    def to_dict(self) -> dict:
        """What was run, for the report (the arms are already expanded)."""
        return {"name": self.name, "checkpoint": self.checkpoint, "dataset": self.dataset, "train_args": list(self.train_args), "device": self.device,
                "arms": [{"name": a.name, "args": list(a.args), "note": a.note} for a in self.arms],
                "grid": {k: list(v) if isinstance(v, tuple) else v for k, v in self.grid.__dict__.items()}, "note": self.note}

    def train_command(self, arm: Arm, folder: Path, python: str) -> list[str]:
        return [python, "-m", "next.run", "sdxl-lora", "--models", self.models, "--datasets", self.datasets, "--file", self.checkpoint, "--dataset", self.dataset,
                "--output", str(folder), "--name", "lora", *self.train_args, *arm.args]


TEMPLATE = {
    "name": "range-ablation",
    "note": "Which part of the timestep range gives the painted-over look? Each arm trains on one range only.",
    "models": "/home/okolenmi/comfy/ComfyUI/models/",
    "datasets": "/home/okolenmi/Desktop/B580-diffusion-training/datasets/",
    "dataset": "my_set",
    "checkpoint": "divA.safetensors",
    "output": "lab-out",
    "device": "xpu",
    "train_args": ["--steps", "600", "--save-every", "600", "--set", "runtime.deterministic=true", "--set", "trainer.probe_every=100", "--set", "trainer.probe_items=4"],
    "images": {"train": "/path/to/training/images", "holdout": "/path/to/images/not/in/training", "count": 6, "size": 768},
    "grid": {"prompts": ["", "a portrait of a woman, detailed face"], "seeds": [1, 2, 3], "strengths": [1.0],
             "steps": 20, "cfg": 6.0, "sampler": "dpmpp_2m", "scheduler": "karras", "width": 1024, "height": 1024},
    "arms": [{"name": "full"},
             {"name": "low-noise", "args": ["--set", "noiser.low=0", "--set", "noiser.high=250"], "note": "t 0-250 only"},
             {"name": "mid-noise", "args": ["--set", "noiser.low=250", "--set", "noiser.high=600"]},
             {"name": "high-noise", "args": ["--set", "noiser.low=600", "--set", "noiser.high=999"]}],
}
