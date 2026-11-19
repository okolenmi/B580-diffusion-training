"""Generation nodes: a whole text-to-image graph runs on the CPU with tiny stand-ins for the weights, and what it writes is reproducible.

    python -m tests.check_generate

Covers the adapter list syntax, the preview pictures a page shows, the starter graph (every wire resolves), the run from prompt to PNG files (shape, seeds,
the recipe inside each file, stopping), and that the same recipe gives the same pixels alone or in a batch.
"""

from __future__ import annotations

import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import starters
from engine.devices import Accelerator
from engine.generate.artifacts import ArtifactFolder
from engine.generate.recipe import ImageStore
from engine.runtime.factory import make_runtime
from graph import GraphRunner, NodeRegistry
from graph.node import Environment
from nodes.generate import decode, encode, sampler
from nodes.model.adapter_stack import parse_stack
from console import NodeTrace
from .pipeline_nodes import TinyFamily
from .support import check, raises

REGISTRY = NodeRegistry(packages=("nodes", "tests.pipeline_nodes"))
SETTINGS = {"device.default": "cpu", "precision.denoiser": "fp32", "precision.encoders": "fp32", "precision.vae": "fp32"}


def graph(tmp: Path, **options):
    options = {"width": 64, "height": 64, "steps": 4, **options}
    return starters.sdxl_image(model="", output=str(tmp / "out"), settings=SETTINGS,
                               source=("TinyModelSource", {"file": str(tmp / "model.safetensors")}), **options)


class Trace:
    def __init__(self, stop_after: int | None = None, cancel: threading.Event | None = None):
        self.frames: list[tuple[str, dict]] = []
        self.stop_after, self.cancel = stop_after, cancel

    def report(self, stream, frame):
        self.frames.append((stream, frame))
        if self.stop_after and frame.get("step", 0) >= self.stop_after and stream.startswith("node:"):
            self.cancel.set()


def execute(doc, stop_after: int | None = None, artifacts: Path | None = None):
    cancel = threading.Event()
    trace = Trace(stop_after, cancel)
    env = Environment(services={"artifacts": ArtifactFolder(artifacts)} if artifacts else {})
    runner = GraphRunner(REGISTRY, reporter=trace, env=env, runtime_factory=make_runtime, observer=NodeTrace(doc, Accelerator("cpu"), None))
    with ExitStack() as stack:
        for module in (encode, sampler, decode):
            stack.enter_context(mock.patch.object(module, "get_family", lambda _id: FAMILY))
        validation = runner.validate(doc)
        outcome = runner.run(doc, cancel)
    return validation, outcome, trace


FAMILY = TinyFamily()


def adapter_list() -> None:
    check(parse_stack("a.safetensors\nb.safetensors=0.5\n# note\n\nc.safetensors=off\n") == [("a.safetensors", 1.0), ("b.safetensors", 0.5)],
          "a bare name is strength 1, =off and comments are left out")
    check(parse_stack("x=-0.5") == [("x", -0.5)], "a negative strength is allowed")
    raises(ValueError, lambda: parse_stack("x=loud"), "a strength that is not a number is refused", "line 1")
    raises(ValueError, lambda: parse_stack("=0.5"), "a line without a file name is refused", "no file name")


def run_graph() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        doc = graph(tmp, prompt="a red fox", negative="blurry", seed=7, batch=2)
        validation, outcome, trace = execute(doc, artifacts=tmp / "art")
        check(validation.ok, f"the starter graph validates ({[i.message for i in validation.errors][:2]})")
        check(outcome.ok and not outcome.stopped, f"the run finishes ({outcome.error})")
        files = sorted((tmp / "out").glob("*.png"))
        check(len(files) == 2, f"one PNG per image ({len(files)})")
        metas = [ImageStore.read_meta(f) for f in files]
        check(sorted(m["recipe"]["seed"] for m in metas) == [7, 8] and all(m["recipe"]["prompt"] == "a red fox" and m["recipe"]["steps"] == 4 for m in metas),
              "every file carries its recipe: image i of a batch has seed + i")
        from PIL import Image
        check(all(Image.open(f).size == (64, 64) for f in files), "the images have the size asked for")
        steps = [f["step"] for s, f in trace.frames if s.startswith("node:") and "total" in f and f["total"] == 4]
        check(steps and steps[-1] == 4 and steps == sorted(steps), f"the sampler reports its steps on its own stream ({steps})")

        (tmp / "single").mkdir()
        shown = [f for s, f in trace.frames if s.startswith("node:") and "images" in f]
        listed = shown[-1]["images"] if shown else []
        check(len(listed) == 2 and all((tmp / "art" / i["path"]).is_file() and i["path"].startswith("n_") for i in listed)
              and [i["seed"] for i in listed] == [7, 8],
              f"Preview Image reports its pictures by path under its own node folder ({listed})")

        single = graph(tmp / "single", prompt="a red fox", negative="blurry", seed=8, batch=1)
        _, again, _ = execute(single)
        alone = sorted((tmp / "single" / "out").glob("*.png"))
        check(again.ok and len(alone) == 1 and Image.open(alone[0]).tobytes() == Image.open(files[1]).tobytes(),
              "seed 8 alone gives the pixels it gave as the second image of a batch")

        shutil_out = tmp / "out"
        for f in shutil_out.glob("*.png"):
            f.unlink()
        _, stopped, _ = execute(graph(tmp, prompt="x", steps=50, batch=1), stop_after=2)
        check(stopped.stopped and not list(shutil_out.glob("*.png")), f"a stop request ends sampling and writes nothing ({stopped.error})")


def preview_housekeeping() -> None:
    with tempfile.TemporaryDirectory() as raw:
        from engine.generate.recipe import Image, Recipe
        import torch
        folder = ArtifactFolder(raw)
        folder.KEEP = 3
        for seed in range(6):
            folder.put("n_aaaaaa", [Image(Recipe(prompt="p", seed=seed), torch.zeros(3, 8, 8, dtype=torch.uint8), {})])
        left = len(list((Path(raw) / "n_aaaaaa").glob("*.png")))
        check(left == 3, f"only the newest few pictures of a node are kept ({left})")


def refusals() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        _, outcome, _ = execute(graph(tmp, prompt="x", width=68))
        check(not outcome.ok and "multiple of 8" in str(outcome.error), f"a size that is not a multiple of 8 is refused ({outcome.error})")


if __name__ == "__main__":
    adapter_list()
    run_graph()
    preview_housekeeping()
    refusals()
    print("\ndone")
