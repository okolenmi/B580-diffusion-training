"""The lab's setup file: where your models, datasets, images and saved adapters are, written once and used by every template.

    python -m next.lab setup > lab-setup.json        (fill it in once)
    python -m next.lab template regions               (reads ./lab-setup.json, or --setup FILE)

A template can name a saved adapter as ``{lora:NAME}``; NAME is a key of ``loras`` in the setup file. A name the file does not have,
or a value that is still a placeholder, stops the command instead of producing a file that would run empty.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_FILE = "lab-setup.json"
PLACEHOLDER = "/path/to"
REFERENCE = re.compile(r"\{lora:([^{}]+)\}")
REQUIRED = ("models", "datasets", "dataset", "file")
FLAGS = {"models": "--models", "datasets": "--datasets", "dataset": "--dataset", "file": "--file", "resolution": "--resolution"}

SKELETON = {
    "models": f"{PLACEHOLDER}/ComfyUI/models/",
    "datasets": f"{PLACEHOLDER}/datasets/",
    "dataset": "my_set",
    "file": "checkpoint.safetensors",
    "resolution": 768,
    "output": "lab-out",
    "device": "xpu",
    "images": {"train": f"{PLACEHOLDER}/training/images", "holdout": f"{PLACEHOLDER}/images/not/in/training", "count": 6, "size": 768},
    "loras": {"best": f"{PLACEHOLDER}/lora_step000600.safetensors"},
}


class SetupError(ValueError):
    """The setup file or a template cannot be used as it is."""


def skeleton() -> dict:
    return json.loads(json.dumps(SKELETON))


def load(path: str | Path | None) -> dict:
    """The setup file at ``path``, or ``./lab-setup.json`` when none is given."""
    target = Path(path or DEFAULT_FILE)
    if not target.exists():
        raise SetupError(f"no setup file {target}: run `python -m next.lab setup > {DEFAULT_FILE}` and fill it in (or give --setup FILE)")
    try:
        return json.loads(target.read_text(encoding="utf8"))
    except json.JSONDecodeError as error:
        raise SetupError(f"{target} is not valid JSON: {error}") from error


def unfilled(value: object, where: str) -> list[str]:
    """Where a string still holds the placeholder path."""
    if isinstance(value, str):
        return [where] if PLACEHOLDER in value else []
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in unfilled(v, f"{where}.{k}")]
    return []


def apply(template: dict, setup: dict) -> dict:
    """``template`` with the setup written into it: models, datasets, dataset, checkpoint and resolution in the training command; images,
    grid, output and device as fields; ``{lora:NAME}`` replaced. Arms, steps and notes stay as the template has them."""
    missing = [k for k in REQUIRED if not setup.get(k)]
    if missing:
        raise SetupError(f"the setup file has no value for: {', '.join(missing)}")
    wanted = set(REFERENCE.findall(json.dumps(template)))
    loras = setup.get("loras", {})
    absent = sorted(wanted - set(loras))
    if absent:
        raise SetupError(f"this template uses the saved adapter(s) {', '.join(absent)}: add them to `loras` in the setup file")
    problems = unfilled({k: setup[k] for k in (*REQUIRED, "images") if k in setup}, "setup") + unfilled({n: loras[n] for n in wanted}, "setup.loras")
    if problems:
        raise SetupError("still placeholders in the setup file: " + ", ".join(problems))
    out = dict(template)
    for key in ("images", "grid", "output", "device"):
        if key in setup:
            out[key] = setup[key]
    command = out.get("train_command", "")
    for key, flag in FLAGS.items():
        if key in setup:
            pattern = rf"(?<=\s){re.escape(flag)}\s+\S+"
            if re.search(pattern, command):
                command = re.sub(pattern, lambda _: f"{flag} {setup[key]}", command, count=1)
    out["train_command"] = command
    return _fill(out, loras)


def _fill(value, loras: dict):
    if isinstance(value, str):
        return REFERENCE.sub(lambda m: str(loras[m.group(1)]), value)
    if isinstance(value, list):
        return [_fill(v, loras) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, loras) for k, v in value.items()}
    return value


def check_experiment(experiment) -> list[str]:
    """What would make a run fail or come out empty, said before anything starts: placeholder paths, adapter files and image folders that are not there."""
    found = []
    for where, value in (("models", experiment.models), ("datasets", experiment.datasets), ("training images", experiment.train_images), ("holdout images", experiment.holdout_images)):
        if PLACEHOLDER in value:
            found.append(f"{where}: {value} is still a placeholder")
        elif value and not Path(value).exists():
            found.append(f"{where}: {value} does not exist")
    for arm in experiment.arms:
        if arm.lora and not Path(arm.lora).is_file():
            found.append(f"arm {arm.name}: the adapter file {arm.lora} does not exist" + (" (a placeholder)" if PLACEHOLDER in arm.lora else ""))
        if any(PLACEHOLDER in a for a in arm.args):
            found.append(f"arm {arm.name}: an argument is still a placeholder")
    return found
