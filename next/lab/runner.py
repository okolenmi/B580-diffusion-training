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

import gc
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from ..engine.devices import Accelerator
from ..families.general.adapters.schemes import read_adapter
from .experiment import Arm, Experiment
from .render import ImageStore, Renderer

LOSS_LINE = re.compile(r"^step (\d+)/(\d+)\s+loss ([0-9.eE+-]+)(?:\s+(\d+) ms)?")
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
        """Trains the arm. An earlier session of the same arm (a power cut, a stop) left its console in ``train.log`` and its training states
        in ``train/resume``: the process continues from the newest state, the console is appended to, and the steps a cut made it repeat
        count once (the last line of a step wins)."""
        self.folder.mkdir(parents=True, exist_ok=True)
        output = (self.folder / "train").resolve()
        command = self._command(self.arm, output, self.python)
        log_path = self.folder / "train.log"
        resumed = log_path.exists()
        steps: dict[int, tuple[float, str, int]] = {}  # step -> (loss, console line, milliseconds)
        probes: dict[int, dict] = {}
        if resumed:
            self._read(log_path.read_text(encoding="utf8", errors="replace").splitlines(), steps, probes)
            say(f"  {self.arm.name}: continuing an earlier session ({len(steps)} steps in its log)")
        started = time.monotonic()
        with log_path.open("a" if resumed else "w", encoding="utf8") as log:
            log.write(("\n--- resumed ---\n" if resumed else "") + "$ " + " ".join(command) + "\n")
            process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf8", errors="replace")
            for line in process.stdout:
                log.write(line)
                log.flush()
                found = self._read([line.rstrip("\n")], steps, probes)
                if found is not None and found % WINDOW == 0:
                    window = [steps[i][0] for i in range(found - WINDOW + 1, found + 1) if i in steps]
                    say(f"  {self.arm.name}: mean loss of steps {found - WINDOW + 1}-{found} {LossWindows.mean(window):.4f}")
            process.wait()
        windows = LossWindows()
        for step in sorted(steps):
            windows.add(steps[step][0], steps[step][1])
        lora = newest_adapter(output)
        seconds = sum(ms for _, _, ms in steps.values()) / 1000 if resumed else time.monotonic() - started  # a resumed arm's wall time includes the gaps
        result = Training(str(lora) if lora else "", max(steps, default=0), seconds, windows.all, [probes[k] for k in sorted(probes)], process.returncode, windows.last())
        (self.folder / "train.json").write_text(json.dumps({"command": command, **result.to_dict(), "resumed": resumed}, indent=1), encoding="utf8")
        if process.returncode == 0 and lora:
            shutil.rmtree(output / "resume", ignore_errors=True)  # the result is safe; the states were only for a cut
        return result

    @staticmethod
    def _read(lines: list[str], steps: dict, probes: dict) -> int | None:
        """Folds console lines into the per-step records; returns the step of the last loss line among them."""
        latest = max(steps, default=0)
        found = None
        for line in lines:
            if match := LOSS_LINE.match(line):
                found = latest = int(match.group(1))
                steps[latest] = (float(match.group(3)), line, int(match.group(4) or 0))
            elif match := PROBE_LINE.match(line):
                thirds = [float(v) for v in match.group(1).split("/")]
                probes[latest] = {"step": latest, "low": thirds[0], "mid": thirds[1], "high": thirds[2], "worst": float(match.group(2))}
        return found


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
        for warning in self.warnings():
            self.say(warning)

    MIN_FREE_GB = 20

    def warnings(self) -> list[str]:
        """Things in the experiment file or the folder that are probably not meant: arms that are the same run, and results of other
        arms left in this experiment's folder (the report lists only the arms of the file)."""
        twins: dict[tuple, list[str]] = {}
        for arm in self.experiment.arms:
            twins.setdefault((arm.args, arm.lora, arm.reuse, arm.ema, arm.eps_scale), []).append(arm.name)
        found = [f"arms {', '.join(names)} have the same arguments: they are repeats of one run (fine as a noise control, but is it a copy-paste slip?)"
                 for names in twins.values() if len(names) > 1]
        free = shutil.disk_usage(self.root).free / 2**30
        if free < self.MIN_FREE_GB:
            found.append(f"only {free:.1f} GB are free on the disk of {self.root}; a full disk makes saves and pictures fail half-way (free some space, or ignore this if the run is small)")
        known = {arm.name for arm in self.experiment.arms}
        others = sorted(p.name for p in self.root.iterdir() if p.is_dir() and p.name not in known and (p / "train.json").exists())
        if others:
            found.append(f"{self.root} also holds results of arms that are not in this file: {', '.join(others)}. They are kept but are not in the report; "
                         f"to compare them, add arms with \"lora\": the adapter file in <arm>/train, or give this file another name")
        return found

    def folder(self, arm: Arm | str) -> Path:
        return self.root / (arm if isinstance(arm, str) else arm.name)

    def train(self, arms: list[Arm], force: bool = False) -> None:
        for arm in arms:
            record = self.folder(arm) / "train.json"
            if arm.reuse:
                self._borrow(arm)
                continue
            if arm.lora:
                self._adopt(arm)
                continue
            if record.exists() and not force and json.loads(record.read_text())["lora"]:
                self.say(f"{arm.name}: trained already")
                continue
            if force:  # not a continuation: forget the earlier session
                shutil.rmtree(self.folder(arm) / "train" / "resume", ignore_errors=True)
                (self.folder(arm) / "train.log").unlink(missing_ok=True)
            self.say(f"{arm.name}: training ({' '.join(arm.args) or 'no extra arguments'})")
            result = self._train(self.experiment, arm, self.folder(arm))
            if result.returncode or not result.lora:
                tail = (self.folder(arm) / "train.log").read_text(encoding="utf8", errors="replace").splitlines()[-8:]
                self.say(f"{arm.name}: training failed (code {result.returncode}); the end of {self.folder(arm) / 'train.log'}:\n    " + "\n    ".join(tail))
                if not any((self.folder(a) / "train.json").exists() and json.loads((self.folder(a) / "train.json").read_text())["lora"] for a in self.experiment.arms):
                    raise SystemExit("no arm has trained yet, so the arguments are probably wrong for this machine (memory, paths): fix them and run again")
            else:
                self.say(f"{arm.name}: {result.steps} steps, {result.seconds / 60:.1f} min, {LossWindows.mean(result.losses[-WINDOW:]):.4f} mean loss of the last {min(WINDOW, len(result.losses))} steps")

    def _adopt(self, arm: Arm) -> None:
        """An arm with an adapter file of its own is rendered as it is; its training record says where the file is."""
        if not Path(arm.lora).exists():
            self.say(f"{arm.name}: the adapter file {arm.lora} does not exist")
            return
        self.folder(arm).mkdir(parents=True, exist_ok=True)
        (self.folder(arm) / "train.json").write_text(json.dumps({"command": [], "lora": str(Path(arm.lora).resolve()), "steps": 0, "seconds": 0.0, "losses": [], "probes": [], "returncode": 0,
                                                                 "recent": None, "adopted": True}, indent=1), encoding="utf8")
        self.say(f"{arm.name}: renders {arm.lora}")

    def _borrow(self, arm: Arm) -> None:
        """An arm that renders another arm's adapter gets a copy of that arm's training record; with ``ema`` the record points at the averaged file beside it."""
        source = self.folder(arm.reuse) / "train.json"
        record = json.loads(source.read_text()) if source.exists() else {}
        if not record.get("lora"):
            self.say(f"{arm.name}: {arm.reuse} has no adapter yet, so there is nothing to reuse")
            return
        if arm.ema:
            lora = Path(record["lora"])
            averaged = lora.with_name(f"{lora.stem}_ema{arm.ema:g}{lora.suffix}")
            if not averaged.exists():
                self.say(f"{arm.name}: {averaged.name} is not next to the adapter of {arm.reuse}: that arm was trained without `--ema {arm.ema:g}`")
                return
            record = {**record, "lora": str(averaged), "ema": arm.ema}
        self.folder(arm).mkdir(parents=True, exist_ok=True)
        (self.folder(arm) / "train.json").write_text(json.dumps({**record, "reused": arm.reuse}, indent=1), encoding="utf8")
        self.say(f"{arm.name}: uses the {'averaged ' if arm.ema else ''}adapter of {arm.reuse}")

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
        recipes = [replace(r, eps_scale=arm.eps_scale) if arm is not None else r for r in exp.grid.recipes(strengths)]
        started = time.monotonic()
        count = 0

        def keep(image) -> None:
            nonlocal count
            store.save(image)
            count += 1
            self.say(f"  image {count}/{len(recipes)} saved, {(time.monotonic() - started) / count:.0f} s per image including the decoder")

        Renderer(backend, device=exp.device).render(recipes, self.meta(arm, lora), on_image=keep)
        if arm is not None and (exp.train_images or exp.holdout_images):
            source = self.folder(arm.reuse) / "timestep_loss.json" if arm.reuse and not arm.ema else None
            if source is not None and source.exists():
                (folder / "timestep_loss.json").write_text(source.read_text(), encoding="utf8")  # the same adapter: the same table
            else:
                (folder / "timestep_loss.json").write_text(json.dumps(self._timestep_loss(backend), indent=1), encoding="utf8")
        del backend
        gc.collect()
        Accelerator(exp.device).empty_cache()

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
