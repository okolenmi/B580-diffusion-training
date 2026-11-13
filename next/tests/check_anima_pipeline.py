"""Acceptance: the Anima starter graph runs on the CPU from dataset to saved LoRA, with the real Anima objective, conditioning kit,
adapter layout and file schemes around tiny stand-in networks.

    python -m next.tests.check_anima_pipeline
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from .. import starters
from ..data.store import DatasetStore
from .check_data import make_images
from .check_pipeline import SETTINGS, execute
from .pipeline_nodes import TinyAnimaFamily
from .support import check, run


def graph(tmp: Path, steps: int):
    return starters.anima_lora(model="", dataset="toy", output=str(tmp / "out"), name="toy", resolution=64, batch_size=2, steps=steps,
                               lr=1e-3, rank=4, save_every=2, settings=SETTINGS, source=("TinyAnimaSource", {"file": str(tmp / "model.safetensors")}),
                               overrides={"latents": {"batch_size": 2}, "buckets": {"multiple": 1}, "batcher": {"seed": 3}})


def anima_run():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        folder = tmp / "images"
        folder.mkdir()
        make_images(folder)
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(folder)
        library.close()
        validation, outcome, trace = execute(store, graph(tmp, 4), family=TinyAnimaFamily)
        check(validation.ok, f"the graph validates ({[i.message for i in validation.errors][:2]})")
        check(outcome.ok and not outcome.stopped, f"the run finishes ({outcome.error})")
        losses = [f["loss"] for st, f in trace.frames if st.startswith("node:") and "loss" in f and "kind" not in f]
        check(len(losses) == 4 and all(torch.isfinite(torch.tensor(losses))), f"four finite losses ({losses})")
        saved = sorted(p.name for p in (tmp / "out").glob("*.safetensors"))
        check(saved == ["toy_step000002.safetensors", "toy_step000004.safetensors"], f"a file every 2 steps ({saved})")
        state, meta = TinyAnimaFamily().read_adapter(str(tmp / "out" / "toy_step000004.safetensors"))
        names = [k for k in state if k.endswith(".lora_down.weight")]
        check(names and all(k.startswith("lora_unet_blocks_") for k in names) and not any("llm_adapter" in k for k in state),
              f"a kohya LoRA over the blocks, none over the text adapter ({len(names)} layers)")
        check(any(float(v.abs().sum()) > 0 for k, v in state.items() if k.endswith(".lora_up.weight")), "the up matrices moved away from zero")


if __name__ == "__main__":
    run(anima_run)
