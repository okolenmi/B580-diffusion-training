"""Acceptance: a run that is cut and started again in a new process ends where the run that was never cut ends.

    python -m next.tests.check_resume

The recipes are the ones the lab trains with (three timestep regions, a ladder, warmup, LoRA+, the gradient-aligned start, the
anchor loss), on the tiny stand-in model and the CPU. Each recipe is trained twice: straight through, and as a first process that
stops at step CUT followed by a second process that continues to the end. The adapters are compared weight by weight and so are
the losses of every step, so a part of the training state that a resume forgets (the data position, the ladder, the warmup, the
optimizer's moments, the anchor's picks) shows up as a difference.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ..data.store import DatasetStore
from .check_data import make_images
from .support import check, run

STEPS, CUT = 8, 4
BASE = dict(warmup=3, lr=2e-3, lora_plus=4.0, resume_every=CUT, save_every=1000)
RECIPES = {
    "plain": dict(BASE),  # control: nothing but the optimizer state to carry over
    "regions + ladder + warmup + LoRA+": dict(BASE, regions_from=5, regions_span=4, overrides={"noiser": {"regions": 3, "layout": "strata"}}),
    "aligned start": dict(BASE, aligned_init=2),
    "anchor": dict(BASE, anchor=1.0, anchor_every=2),
    "batch ramp": dict(BASE, batch_first=1, batch_first_steps=5),
}


def child(tmp: Path, recipe: str, steps: int, out: Path) -> None:
    """One training process: trains ``steps`` steps of ``recipe`` in ``out``, writes every step's metrics beside the adapter."""
    from . import check_pipeline as pipeline

    options = dict(RECIPES[recipe])
    overrides = options.pop("overrides", {})
    doc = pipeline.starters.sdxl_lora(model="", dataset="toy", output=str(out), name="toy", resolution=64, batch_size=2, steps=steps, rank=4,
                                      settings={**pipeline.SETTINGS}, source=("TinyModelSource", {"file": str(tmp / "model.safetensors")}),
                                      overrides={"latents": {"batch_size": 2}, "buckets": {"multiple": 1}, "batcher": {"seed": 3}, **overrides}, **options)
    _, outcome, trace = pipeline.execute(DatasetStore(tmp / "datasets"), doc)
    steps_seen = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
    (out / "metrics.json").write_text(json.dumps({"ok": outcome.ok, "error": str(outcome.error), "steps": steps_seen}))


def launch(tmp: Path, recipe: str, steps: int, out: Path) -> dict:
    done = subprocess.run([sys.executable, "-m", "next.tests.check_resume", "--child", str(tmp), recipe, str(steps), str(out)],
                          capture_output=True, text=True, check=False)
    if done.returncode:
        print(done.stderr[-2000:])
    return json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {"ok": False, "error": done.stderr[-300:], "steps": []}


def final(out: Path, step: int):
    from .pipeline_nodes import TinyFamily

    return TinyFamily().read_adapter(str(out / f"toy_step{step:06d}.safetensors"))[0]


def compare() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        (tmp / "images").mkdir()
        make_images(tmp / "images")
        library = DatasetStore(tmp / "datasets").library("toy", create=True)
        library.ingest(tmp / "images")
        library.close()
        for recipe in RECIPES:
            whole, first, second = tmp / "whole", tmp / "first", tmp / "second"
            for folder in (whole, first, second):
                shutil.rmtree(folder, ignore_errors=True)
            a = launch(tmp, recipe, STEPS, whole)
            b = launch(tmp, recipe, CUT, first)
            shutil.copytree(first / "resume", second / "resume")  # the cut left only its states
            c = launch(tmp, recipe, STEPS, second)
            check(a["ok"] and b["ok"] and c["ok"] and len(c["steps"]) == STEPS - CUT, f"{recipe}: three processes ran ({a['error']}{b['error']}{c['error']}, {len(c['steps'])} steps after the cut)")
            if not (a["ok"] and c["ok"]):
                continue
            wa, wc = final(whole, STEPS), final(second, STEPS)
            worst = max(float((wa[k] - wc[k]).abs().max()) for k in wa) if wa.keys() == wc.keys() else float("inf")
            check(worst < 1e-5, f"{recipe}: the continued adapter equals the uninterrupted one (largest weight difference {worst:.2e})")
            lost = max(abs(x["loss"] - y["loss"]) for x, y in zip(a["steps"][CUT:], c["steps"]))
            check(lost < 1e-5, f"{recipe}: the losses of the steps after the cut equal the uninterrupted ones (largest difference {lost:.2e})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5]))
    else:
        run(compare)
