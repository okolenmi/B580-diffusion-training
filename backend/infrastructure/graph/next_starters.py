"""NextStarters -- the GraphStarters port over ``next/starters``.

The form of a starter is the few choices a person makes before the graph exists (what to train on, how long,
how fast); every other option stays a parameter of a node, edited on the canvas where it has its own help.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ...application.errors import InvalidQueryError, StarterNotFoundError
from ...application.ports.graph_starters import GraphStarters
from . import tree

_COMMON = (
    # name, label, type, picker/choices, doc
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


@dataclass(frozen=True)
class _Starter:
    name: str
    label: str
    doc: str
    fields: tuple
    required: tuple[str, ...]
    options_class: type
    build: Callable[..., dict]


_SDXL = _Starter(
    "sdxl-lora", "SDXL LoRA training",
    "Dataset, latent cache, prompts, buckets, conditioner, adapter, optimizer, trainer and saver, ready to run.",
    (("model", "Checkpoint", "string", {"picker": "model_file", "scope": "checkpoints"}, "The SDXL checkpoint to train on."), *_COMMON),
    ("model", "dataset", "output"), tree.SdxlLoraOptions, lambda values: tree.sdxl_lora(**values).as_dict())

_ANIMA = _Starter(
    "anima-lora", "Anima LoRA training",
    "The same graph for Anima: its DiT, Qwen3 text encoder and VAE as three Load Model nodes, flow-matching noiser, 16 px latent grid.",
    (("model", "DiT", "string", {"picker": "model_file", "scope": "diffusion_models"}, "The Anima transformer file (it holds the text adapter)."),
     ("text_encoder", "Text encoder", "string", {"picker": "model_file", "scope": "text_encoders"}, "The Qwen3-0.6B file."),
     ("vae", "VAE", "string", {"picker": "model_file", "scope": "vae"}, "The Wan 2.1 / Qwen-Image VAE file."),
     *_COMMON,
     ("sampler", "Timestep sampler", "string", {"choices": ["sigmoid", "logit_normal", "uniform", "cosmap"]}, "How training timesteps are drawn."),
     ("shift", "Shift", "float", {}, "Pushes timesteps toward noise; empty is the family's 3.0.")),
    ("model", "text_encoder", "vae", "dataset", "output"), tree.AnimaLoraOptions, lambda values: tree.anima_lora(**values).as_dict())


class NextStarters(GraphStarters):
    def __init__(self) -> None:
        self._starters = {starter.name: starter for starter in (_SDXL, _ANIMA)}

    def starters(self) -> list[dict]:
        listed = []
        for starter in self._starters.values():
            defaults = starter.options_class()
            options = []
            for name, label, kind, extra, doc in starter.fields:
                entry = {"name": name, "label": label, "type": kind, "doc": doc, "required": name in starter.required, "advanced": False, **extra}
                if getattr(defaults, name, None) is not None:
                    entry["default"] = getattr(defaults, name)
                options.append(entry)
            listed.append({"name": starter.name, "label": starter.label, "doc": starter.doc, "options": options})
        return listed

    def build(self, name: str, values: dict) -> dict:
        starter = self._starters.get(name)
        if starter is None:
            raise StarterNotFoundError(f"no starter named {name!r}")
        known = {field[0] for field in starter.fields}
        stray = sorted(set(values) - known)
        if stray:
            raise InvalidQueryError(f"unknown options: {', '.join(stray)}")
        kept = {key: value for key, value in values.items() if value not in ("", None) or key in starter.required}
        missing = [field for field in starter.required if not str(kept.get(field, "")).strip()]
        if missing:
            raise InvalidQueryError(f"the starter needs: {', '.join(missing)}")
        try:
            return starter.build(kept)
        except (TypeError, ValueError) as err:
            raise InvalidQueryError(f"the starter refused these values: {err}") from err
