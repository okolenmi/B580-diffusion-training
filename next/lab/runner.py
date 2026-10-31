"""Running an experiment: train each arm in its own process, render the grid with each result, and keep everything where the report can find it.

    <output>/<experiment>/
        experiment.json          what was run
        reference/images/        the grid rendered without any adapter
        <arm>/train.log, train.json, train/lora_stepNNNNNN.safetensors
        <arm>/images/*.png, index.json      every image carries its recipe, the arm and the adapter
        <arm>/timestep_loss.json
        report.md, sheets/
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..families.general.adapters.schemes import read_adapter
from .experiment import Arm, Experiment
from .render import ImageStore, Renderer

LOSS_LINE = re.compile(r"^step (\d+)/(\d+)\s+loss ([0-9.eE+-]+)")
THIRDS_LINE = re.compile(r"by t((?: (?:low|mid|high) [0-9.eE+-]+)+)")
WINDOW = 100  # steps one reported loss averages over
PROBE_LINE = re.compile(r"^\s+probe: adapter/base loss by third ([0-9./]+), worst ([0-9.]+)")
LORA_FILE = re.compile(r"^lora_step(\d+)\.safetensors$")
REPO_ROOT = Path(__file__).resolve().parents[2]


class LossWindows:
    """Losses of single steps mean little (each is one random timestep); what is reported is the mean over a window of steps,
    overall and by third of the timestep range."""

    def __init__(self) -> None:
        self.all: list[float] = []
        self.thirds: dict[str, list[tuple[int, float]]] = {"low": [], "mid": [], "high": []}  # (step index, loss) of the steps that drew from the third

    def add(self, loss: float, line: str) -> None:
        self.all.append(loss)
        if found := THIRDS_LINE.search(line):
            parts = found.group(1).split()
            for name, value in zip(parts[::2], parts[1::2]):
                self.thirds[name].append((len(self.all), float(value)))

    @staticmethod
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    def last(self, count: int = WINDOW) -> dict[str, float]:
        """Mean over the last ``count`` steps; a third counts only the steps that drew from it."""
        return {"loss": self.mean(self.all[-count:]), **{k: self.mean([x for i, x in v if i > len(self.all) - count]) for k, v in self.thirds.items()}}

    def describe(self, first: int, last: int) -> str:
        window = self.all[first - 1:last]
        return f"mean loss of steps {first}-{last} {self.mean(window):.4f}"


@dataclass
class Training:
    lora: str
    steps: int
    seconds: float
    losses: list[float]
    probes: list[dict]
    returncode: int
    recent: dict | None = None  # mean loss over the last WINDOW steps, overall and by third of the range

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def newest_adapter(folder: Path) -> Path | None:
    found = [(int(m.group(1)), p) for p in folder.glob("lora_step*.safetensors") if (m := LORA_FILE.match(p.name))]
    return max(found)[1] if found else None


class ArmTraining:
    """One training process; its console is the arm's ``train.log`` and the loss and probe lines are read from it as they come."""

    def __init__(self, experiment: Experiment, arm: Arm, folder: Path, python: str = sys.executable, command: Callable[..., list[str]] | None = None):
        self.experiment, self.arm, self.folder, self.python = experiment, arm, folder, python
        self._command = command or experiment.train_command

    def run(self, say: Callable[[str], None]) -> Training:
        self.folder.mkdir(parents=True, exist_ok=True)
        output = (self.folder / "train").resolve()
        command = self._command(self.arm, output, self.python)
        windows = LossWindows()
        losses = windows.all
        probes: list[dict] = []
        step = 0
        started = time.monotonic()
        with (self.folder / "train.log").open("w", encoding="utf8") as log:
            log.write("$ " + " ".join(command) + "\n")
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace")
            for line in process.stdout:
                log.write(line)
                if match := LOSS_LINE.match(line):
                    step = int(match.group(1))
                    windows.add(float(match.group(3)), line)
                    if step % WINDOW == 0:
                        say(f"  {self.arm.name}: {windows.describe(step - WINDOW + 1, step)} (of {match.group(2)} steps)")
                elif match := PROBE_LINE.match(line):
                    thirds = [float(v) for v in match.group(1).split("/")]
                    probes.append({"step": step, "low": thirds[0], "mid": thirds[1], "high": thirds[2], "worst": float(match.group(2))})
            process.wait()
        lora = newest_adapter(output)
        result = Training(str(lora) if lora else "", step, time.monotonic() - started, losses, probes, process.returncode, windows.last())
        (self.folder / "train.json").write_text(json.dumps({"command": command, **result.to_dict()}, indent=1), encoding="utf8")
        return result


def adapter_facts(path: str) -> dict:
    """What identifies an adapter file: its name, size, hash and the header the trainer wrote."""
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    _, header = read_adapter(path)
    return {"file": Path(path).name, "mb": round(Path(path).stat().st_size / 1e6, 1), "sha1": digest.hexdigest(), **{k: header.get(k) for k in ("step", "rank", "alpha", "scheme")}}


class Lab:
    """``backend_factory(lora_path_or_None)`` builds the rendering backend (SDXL in real runs, a stand-in in tests)."""

    def __init__(self, experiment: Experiment, backend_factory: Callable[[str | None], object], say: Callable[[str], None] = print,
                 trainer: Callable[[Experiment, Arm, Path], Training] | None = None):
        self.experiment, self.backend_factory, self.say = experiment, backend_factory, say
        self._train = trainer or (lambda exp, arm, folder: ArmTraining(exp, arm, folder).run(self.say))
        self.root = experiment.root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "experiment.json").write_text(json.dumps(experiment.to_dict(), indent=1), encoding="utf8")

    def folder(self, arm: Arm | str) -> Path:
        return self.root / (arm if isinstance(arm, str) else arm.name)

    def train(self, arms: list[Arm], force: bool = False) -> None:
        for arm in arms:
            record = self.folder(arm) / "train.json"
            if record.exists() and not force and json.loads(record.read_text())["lora"]:
                self.say(f"{arm.name}: trained already")
                continue
            self.say(f"{arm.name}: training ({' '.join(arm.args) or 'no extra arguments'})")
            result = self._train(self.experiment, arm, self.folder(arm))
            if result.returncode or not result.lora:
                tail = (self.folder(arm) / "train.log").read_text(encoding="utf8", errors="replace").splitlines()[-8:]
                self.say(f"{arm.name}: training failed (code {result.returncode}); the end of {self.folder(arm) / 'train.log'}:\n    " + "\n    ".join(tail))
                if not any((self.folder(a) / "train.json").exists() and json.loads((self.folder(a) / "train.json").read_text())["lora"] for a in self.experiment.arms):
                    raise SystemExit("no arm has trained yet, so the arguments are probably wrong for this machine (memory, paths): fix them and run again")
            else:
                self.say(f"{arm.name}: {result.steps} steps, {result.seconds / 60:.1f} min, {LossWindows.mean(result.losses[-WINDOW:]):.4f} mean loss of the last {min(WINDOW, len(result.losses))} steps")

    def meta(self, arm: Arm | None, lora: str | None) -> dict:
        exp = self.experiment
        return {"lab": 1, "experiment": exp.name, "arm": arm.name if arm else "reference", "arm_args": list(arm.args) if arm else [], "train_args": list(exp.train_args),
                "checkpoint": exp.checkpoint, "adapter": adapter_facts(lora) if lora else None, "created": datetime.now().isoformat(timespec="seconds")}

    def render(self, arms: list[Arm], force: bool = False) -> None:
        exp = self.experiment
        if exp.reference and (force or not (self.root / "reference" / "images" / "index.json").exists()):
            self.say("reference: rendering without an adapter")
            self._render_one(None, None, (0.0,))
        for arm in arms:
            if not force and self._rendered(arm):
                self.say(f"{arm.name}: rendered already")
                continue
            lora = (json.loads((self.folder(arm) / "train.json").read_text()) if (self.folder(arm) / "train.json").exists() else {}).get("lora")
            if not lora:
                self.say(f"{arm.name}: no adapter to render")
                continue
            self.say(f"{arm.name}: rendering {len(exp.grid.recipes())} images")
            self._render_one(arm, lora, exp.grid.strengths)

    def _rendered(self, arm: Arm) -> bool:
        wants_loss = bool(self.experiment.train_images or self.experiment.holdout_images)
        return (self.folder(arm) / "images" / "index.json").exists() and (not wants_loss or (self.folder(arm) / "timestep_loss.json").exists())

    def _render_one(self, arm: Arm | None, lora: str | None, strengths: tuple[float, ...]) -> None:
        exp = self.experiment
        folder = self.root / "reference" if arm is None else self.folder(arm)
        backend = self.backend_factory(lora)
        store = ImageStore(folder / "images")
        recipes = exp.grid.recipes(strengths)
        started = time.monotonic()
        count = 0

        def keep(image) -> None:
            nonlocal count
            store.save(image)
            count += 1
            self.say(f"  image {count}/{len(recipes)} saved, {(time.monotonic() - started) / count:.0f} s per image including the decoder")

        Renderer(backend, device=exp.device).render(recipes, self.meta(arm, lora), on_image=keep)
        if arm is not None and (exp.train_images or exp.holdout_images):
            (folder / "timestep_loss.json").write_text(json.dumps(self._timestep_loss(backend), indent=1), encoding="utf8")
        del backend

    def _timestep_loss(self, backend) -> dict:
        from .timestep_loss import DEFAULT_T, TimestepLoss, load_images

        exp = self.experiment
        probe = TimestepLoss(backend, exp.timesteps or DEFAULT_T)
        out = {}
        for name, folder in (("train", exp.train_images), ("holdout", exp.holdout_images)):
            if folder:
                self.say(f"  timestep loss on {name} images")
                result = probe.run(probe.latents(load_images(folder, exp.loss_images, exp.loss_size)))
                out[name] = {str(t): list(v) for t, v in result.items()}
        return out
