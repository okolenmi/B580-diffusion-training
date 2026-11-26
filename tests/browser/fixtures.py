"""Graphs and data the browser checks run on the tiny CPU profile: a toy dataset in the project's store, and the training and generation starters
with the stand-in model source (see ``tiny_profile/sitecustomize.py``)."""

from __future__ import annotations

from pathlib import Path

import starters
from data.store import DatasetStore
from tests.check_data import make_images

from .server import DATASET_PREFIX, ROOT

SETTINGS = {"device.default": "cpu", "precision.denoiser": "fp32", "precision.encoders": "fp32", "precision.autocast": "off"}


def toy_dataset(home: Path, name: str = "toy") -> str:
    """Make a six-image dataset in the project's dataset store; the name carries the prefix that ``TestServer`` cleans up."""
    full = DATASET_PREFIX + name
    folder = home / f"images-{name}"
    folder.mkdir(parents=True, exist_ok=True)
    make_images(folder)
    store = DatasetStore(ROOT / "datasets")
    library = store.library(full, create=True)
    library.ingest(folder)
    library.close()
    return full


def lora_graph(home: Path, dataset: str, *, steps: int = 6, save_every: int = 3, out: str = "out", **extra) -> dict:
    """The SDXL LoRA starter on tiny stand-ins, as the dict the library and the apps store."""
    doc = starters.sdxl_lora(model="", dataset=dataset, output=str(home / out), name="toy", resolution=64, batch_size=2, steps=steps, lr=1e-3, rank=4,
                             save_every=save_every, settings=SETTINGS, source=("TinyModelSource", {"file": str(home / "model.safetensors")}),
                             overrides={"latents": {"batch_size": 2}, "buckets": {"multiple": 1}, "batcher": {"seed": 3}}, **extra)
    return doc.as_dict()


def image_graph(home: Path, *, steps: int = 4, **extra) -> dict:
    doc = starters.sdxl_image(model="", output=str(home / "images"), settings=SETTINGS, width=64, height=64, steps=steps,
                              source=("TinyModelSource", {"file": str(home / "model.safetensors")}), **extra)
    return doc.as_dict()
