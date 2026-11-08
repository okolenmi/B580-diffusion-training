"""NextStarters -- the GraphStarters port over ``next/starters``.

The form of a starter is the few choices a person makes before the graph exists (what to train on, how long,
how fast); every other option stays a parameter of a node, edited on the canvas where it has its own help.
"""

from __future__ import annotations

from ...application.errors import InvalidQueryError, StarterNotFoundError
from ...application.ports.graph_starters import GraphStarters
from . import tree

_FIELDS = (
    # name, label, type, picker/choices, doc
    ("model", "Checkpoint", "string", {"picker": "model_file", "scope": "checkpoints"}, "The SDXL checkpoint to train on."),
    ("dataset", "Dataset", "string", {"picker": "dataset"}, "The dataset folder (images with captions)."),
    ("output", "Output folder", "string", {}, "Where adapters, samples and states are written."),
    ("name", "Adapter name", "string", {}, "File name stem of the saved adapters."),
    ("resolution", "Resolution", "int", {}, "Longest training side in pixels."),
    ("batch_size", "Batch size", "int", {}, "Images per step."),
    ("steps", "Steps", "int", {}, "Optimizer steps."),
    ("lr", "Learning rate", "float", {}, "Peak learning rate."),
    ("rank", "Rank", "int", {}, "LoRA rank."),
    ("schedule", "Schedule", "string", {"choices": ["constant", "cosine", "linear"]}, "Learning-rate schedule."),
    ("save_every", "Save every", "int", {}, "A checkpoint every N steps."),
)
_REQUIRED = ("model", "dataset", "output")


def _sdxl_lora(values: dict) -> dict:
    missing = [name for name in _REQUIRED if not str(values.get(name, "")).strip()]
    if missing:
        raise InvalidQueryError(f"the starter needs: {', '.join(missing)}")
    try:
        return tree.sdxl_lora(**values).as_dict()
    except (TypeError, ValueError) as err:
        raise InvalidQueryError(f"the starter refused these values: {err}") from err


class NextStarters(GraphStarters):
    def __init__(self) -> None:
        self._builders = {"sdxl-lora": _sdxl_lora}

    def starters(self) -> list[dict]:
        defaults = tree.SdxlLoraOptions()
        options = []
        for name, label, kind, extra, doc in _FIELDS:
            entry = {"name": name, "label": label, "type": kind, "doc": doc, "required": name in _REQUIRED, "advanced": False, **extra}
            if hasattr(defaults, name):
                entry["default"] = getattr(defaults, name)
            options.append(entry)
        return [{"name": "sdxl-lora", "label": "SDXL LoRA training", "doc": "Dataset, latent cache, prompts, buckets, conditioner, adapter, optimizer, trainer and saver, ready to run.", "options": options}]

    def build(self, name: str, values: dict) -> dict:
        builder = self._builders.get(name)
        if builder is None:
            raise StarterNotFoundError(f"no starter named {name!r}")
        known = {field[0] for field in _FIELDS}
        stray = sorted(set(values) - known)
        if stray:
            raise InvalidQueryError(f"unknown options: {', '.join(stray)}")
        return builder({key: value for key, value in values.items() if value not in ("", None) or key in _REQUIRED})
