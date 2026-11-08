"""An experiment: one training setup, the variants (arms) to compare, and the images to render and measure for each."""

from __future__ import annotations

import itertools
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .render import Grid


@dataclass(frozen=True)
class Arm:
    """One variant: a name and the extra training arguments that make it differ from the others."""

    name: str
    args: tuple[str, ...] = ()
    note: str = ""
    eps_scale: float = 1.0  # sampling only: the predicted noise is divided by this (1 = unchanged)
    reuse: str = ""  # the name of an earlier arm whose trained adapter this arm renders instead of training its own
    lora: str = ""  # an adapter file made earlier: rendered as it is, nothing is trained
    ema: float = 0.0  # with `reuse`: render the averaged copy of that arm's adapter (the trainer's `--ema` width) instead of the adapter itself
    cut: int = 0  # the lab stops this arm's training process once that step's state is on disk and starts it again, as a power cut would (a multiple of resume_every)

    def describe(self) -> str:
        parts = [*self.args, *([f"adapter {Path(self.lora).name}"] if self.lora else []), *([f"reuse {self.reuse}"] if self.reuse else []), *([f"cut after step {self.cut}"] if self.cut else []), *([f"ema {self.ema:g}"] if self.ema else []), *([f"eps_scale {self.eps_scale:g}"] if self.eps_scale != 1.0 else [])]
        return " ".join(parts) or "-"


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
    resume_every: int = 200  # steps between the training states that let an interrupted arm continue (0 = off)
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
        out = [Arm(a["name"], tuple(a.get("args", ())), a.get("note", ""), float(a.get("eps_scale", 1.0)), a.get("reuse", ""), a.get("lora", ""), float(a.get("ema", 0.0)), int(a.get("cut", 0))) for a in arms]
        if product:
            for combo in itertools.product(*product):
                out.append(Arm("+".join(c["name"] for c in combo), tuple(x for c in combo for x in c.get("args", ())), " ".join(c.get("note", "") for c in combo).strip()))
        names = [a.name for a in out]
        if len(set(names)) != len(names) or not names:
            raise ValueError("an experiment needs at least one arm and every arm needs its own name")
        for i, arm in enumerate(out):
            if arm.eps_scale <= 0 or (arm.reuse and (arm.reuse not in names[:i] or arm.args or arm.lora)) or (arm.lora and arm.args) or arm.ema < 0 or (arm.ema and not arm.reuse):
                raise ValueError(f"arm {arm.name!r}: eps_scale must be positive, `reuse` names an earlier arm and comes without training arguments, and `ema` needs `reuse`")
            if arm.cut and (arm.cut < 0 or arm.reuse or arm.lora):
                raise ValueError(f"arm {arm.name!r}: `cut` is a step number of an arm that trains")
        return tuple(out)

    @staticmethod
    def split_command(command: str) -> dict:
        """A training command as people type it (``python -m next.run sdxl-lora --models ... --steps 600``) into the experiment's own fields:
        models, datasets, dataset and checkpoint are taken out, ``--output`` and ``--name`` are dropped (the lab chooses them), the rest are the shared arguments."""
        tokens = shlex.split(command.replace("\\\n", " "))
        if "sdxl-lora" not in tokens:
            raise ValueError("the training command must be a `python -m next.run sdxl-lora ...` command")
        tokens = tokens[tokens.index("sdxl-lora") + 1:]
        fields = {"--models": "models", "--datasets": "datasets", "--dataset": "dataset", "--file": "checkpoint"}
        found, rest, i = {}, [], 0
        while i < len(tokens):
            token = tokens[i]
            if token in fields and i + 1 < len(tokens):
                found[fields[token]] = tokens[i + 1]
                i += 2
            elif token in ("--output", "--name") and i + 1 < len(tokens):
                i += 2
            else:
                rest.append(token)
                i += 1
        return {**found, "train_args": rest}

    @staticmethod
    def from_dict(spec: dict) -> Experiment:
        if spec.get("train_command"):
            spec = {**Experiment.split_command(spec["train_command"]), **{k: v for k, v in spec.items() if k != "train_command"}}
        grid = dict(spec.get("grid", {}))
        grid["prompts"] = tuple(grid.get("prompts") or ("",))
        for key in ("seeds", "strengths"):
            if key in grid:
                grid[key] = tuple(grid[key])
        images = spec.get("images", {})
        known = {"train_command", "name", "models", "datasets", "dataset", "checkpoint", "output", "arms", "product", "grid", "train_args", "device", "images", "timesteps", "reference", "resume_every", "note"}
        return Experiment(name=spec["name"], models=spec["models"], datasets=spec["datasets"], dataset=spec["dataset"], checkpoint=spec["checkpoint"],
                          output=spec.get("output", "lab-out"), arms=Experiment.expand(spec.get("arms", []), spec.get("product", [])), grid=Grid(**grid),
                          train_args=tuple(spec.get("train_args", ())), device=spec.get("device", "xpu"), train_images=images.get("train", ""), holdout_images=images.get("holdout", ""),
                          loss_images=images.get("count", 6), loss_size=images.get("size", 768), timesteps=tuple(spec.get("timesteps", ())),
                          reference=spec.get("reference", True), resume_every=spec.get("resume_every", 200), note=spec.get("note", ""), extra={k: v for k, v in spec.items() if k not in known})

    @staticmethod
    def load(path: str | Path) -> Experiment:
        return Experiment.from_dict(json.loads(Path(path).read_text(encoding="utf8")))

    def to_dict(self) -> dict:
        """What was run, for the report (the arms are already expanded)."""
        return {"name": self.name, "checkpoint": self.checkpoint, "dataset": self.dataset, "train_args": list(self.train_args), "device": self.device,
                "arms": [{"name": a.name, "args": list(a.args), "note": a.note, "eps_scale": a.eps_scale, "reuse": a.reuse, "lora": a.lora, "ema": a.ema} for a in self.arms],
                "grid": {k: list(v) if isinstance(v, tuple) else v for k, v in self.grid.__dict__.items()}, "resume_every": self.resume_every, "note": self.note}

    def train_command(self, arm: Arm, folder: Path, python: str) -> list[str]:
        return [python, "-m", "next.run", "sdxl-lora", "--models", self.models, "--datasets", self.datasets, "--file", self.checkpoint, "--dataset", self.dataset,
                "--output", str(folder), "--name", "lora", *(["--resume-every", str(self.resume_every)] if self.resume_every else []), *self.train_args, *arm.args]
