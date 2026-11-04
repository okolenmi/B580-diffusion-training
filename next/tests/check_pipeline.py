"""Acceptance: a whole training graph runs through the graph runtime on the CPU, from dataset to saved adapter.

    python -m next.tests.check_pipeline

The graph is the one a person builds in the editor (dataset chain, conditioner, adapter, optimizer, schedule,
trainer, checkpoint trigger, saver) with only the model weights replaced by tiny stand-ins. It proves the nodes
agree with each other and with the runtime: ports, event wiring and order, stopping, and the files that come out.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from .. import starters
from ..data.store import DatasetStore
from ..engine.devices import Accelerator
from ..engine.runtime.factory import make_runtime
from ..console import NodeTrace
from ..graph import GraphDoc, GraphRunner, NodeRegistry
from ..graph.node import Environment
from ..nodes.conditioning import conditioner
from ..nodes.dataset import batching, latents
from ..nodes.model import inject_adapter
from .check_data import make_images
from .pipeline_nodes import TinyFamily
import torch

from .support import check, run

REGISTRY = NodeRegistry(packages=("next.nodes", "next.tests.pipeline_nodes"))
SETTINGS = {"device.default": "cpu", "precision.denoiser": "fp32", "precision.encoders": "fp32", "precision.autocast": "off"}


def graph(tmp: Path, steps: int, captured: bool = False, **extra) -> GraphDoc:
    return starters.sdxl_lora(model="", dataset="toy", output=str(tmp / "out"), name="toy", resolution=64, batch_size=2, steps=steps,
                              lr=extra.pop("lr", 1e-3), rank=4, save_every=extra.pop("save_every", 2), settings=SETTINGS, graph=captured, **extra,
                              source=("TinyModelSource", {"file": str(tmp / "model.safetensors")}),
                              overrides={"latents": {"batch_size": 2}, "buckets": {"multiple": 1}, "batcher": {"seed": 3}})


class Trace:
    """Collects reporter frames; can ask the run to stop once a step has finished."""

    def __init__(self, stop_after: int | None = None, cancel: threading.Event | None = None):
        self.frames: list[tuple[str, dict]] = []
        self.stop_after, self.cancel = stop_after, cancel

    def report(self, stream, frame):
        self.frames.append((stream, frame))
        if self.stop_after and frame.get("name") == "step_finished" and self.steps() >= self.stop_after:
            self.cancel.set()

    def steps(self) -> int:
        return sum(1 for _, f in self.frames if f.get("name") == "step_finished")


def execute(store: DatasetStore, doc: GraphDoc, stop_after: int | None = None):
    cancel = threading.Event()
    trace = Trace(stop_after, cancel)
    node_trace = NodeTrace(doc, Accelerator("cpu"), None)
    runner = GraphRunner(REGISTRY, reporter=trace, env=Environment(services={"datasets": store}), runtime_factory=make_runtime,
                         observer=node_trace)
    with ExitStack() as stack:
        for module in (conditioner, inject_adapter, latents, batching):
            stack.enter_context(mock.patch.object(module, "get_family", lambda _id: TinyFamily()))
        validation = runner.validate(doc)
        outcome = runner.run(doc, cancel)
    return validation, outcome, trace


def full_run():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        classes = {n.class_name for n in graph(tmp, 2, schedule="cosine").nodes}
        check("CosineLRNode" in classes and "ConstantLRNode" not in classes, "the starter can decay the learning rate (cosine) instead of holding it")
        folder = tmp / "images"
        folder.mkdir()
        make_images(folder)
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(folder)
        library.close()

        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            validation, outcome, trace = execute(store, graph(tmp, steps=4))
        check(len(printed.getvalue().strip().splitlines()) == len(graph(tmp, steps=4).nodes), "the command line reports every finished node")
        check(validation.ok, f"the graph validates ({[i.message for i in validation.errors][:2]})")
        check(outcome.ok and not outcome.stopped, f"the run finishes ({outcome.error})")
        saved = sorted(p.name for p in (tmp / "out").glob("*.safetensors"))
        check(saved == ["toy_step000002.safetensors", "toy_step000004.safetensors"], f"a file every 2 steps, the last one at the end ({saved})")
        state, meta = TinyFamily().read_adapter(str(tmp / "out" / "toy_step000004.safetensors"))
        check(len(state) > 0 and meta.get("step") in (4, "4") and meta.get("rank") in (4, "4"),
              f"the saved file holds adapter tensors and its metadata ({len(state)} tensors, {meta})")
        check(trace.steps() == 4, f"one step event per optimizer step ({trace.steps()})")

        shutil.rmtree(tmp / "out")
        _, stopped, trace = execute(store, graph(tmp, steps=50), stop_after=3)
        check(stopped.ok and stopped.stopped, f"a stop request ends the run cleanly ({stopped.error})")
        saved = sorted(p.name for p in (tmp / "out").glob("*.safetensors"))
        check(saved == ["toy_step000002.safetensors", "toy_step000003.safetensors"], f"the stopped run still saves where it stopped ({saved})")

        shutil.rmtree(tmp / "out")
        _, graphed, trace = execute(store, graph(tmp, steps=3, captured=True))
        said = [f for s, f in trace.frames if s == "graph"]
        check(graphed.ok and trace.steps() == 3 and said and "cannot capture" in said[0]["why"],
              f"a graph run on a device that cannot capture trains eagerly and says so ({graphed.error}, {said[:1]})")

        shutil.rmtree(tmp / "out")
        _, byepoch, trace = execute(store, graph(tmp, steps=0, epochs=2, save_every=1000, save_every_epoch=1))
        steps = [f["summary"]["metrics"] for s, f in trace.frames if f.get("name") == "step_finished"]
        ends = [i + 1 for i, m in enumerate(steps) if m["epoch_end"]]
        saved = sorted(int(p.stem[-6:]) for p in (tmp / "out").glob("*.safetensors"))
        check(byepoch.ok and len(ends) == 2 and len(steps) == ends[-1] and saved == ends,
              f"epochs=2 ends the run after the second epoch and Every Epoch saves at each epoch's last step ({len(steps)} steps, epoch ends {ends}, saved {saved})")

        shutil.rmtree(tmp / "out")
        _, averaged, _ = execute(store, graph(tmp, steps=4, ema="0.1"))
        saved = sorted(p.name for p in (tmp / "out").glob("*.safetensors"))
        plain, _ = TinyFamily().read_adapter(str(tmp / "out" / "toy_step000004.safetensors"))
        mean, _ = TinyFamily().read_adapter(str(tmp / "out" / "toy_step000004_ema0.1.safetensors"))
        key = next(k for k in plain if "lora" in k.lower() or "up" in k.lower())
        check(averaged.ok and saved == ["toy_step000002.safetensors", "toy_step000002_ema0.1.safetensors", "toy_step000004.safetensors", "toy_step000004_ema0.1.safetensors"]
              and plain.keys() == mean.keys() and any(not torch.equal(plain[k], mean[k]) for k in plain),
              f"Adapter EMA writes an averaged file beside each saved one, different from the trained weights ({averaged.error}, {saved})")

        shutil.rmtree(tmp / "out")
        _, aligned, _ = execute(store, graph(tmp, steps=3, aligned_init=2))
        saved = sorted(p.name for p in (tmp / "out").glob("*.safetensors"))
        check(aligned.ok and saved == ["toy_step000002.safetensors", "toy_step000003.safetensors"],
              f"a run with the gradient-aligned initialisation trains and saves as usual ({aligned.error}, {saved})")

        shutil.rmtree(tmp / "out")
        _, whole, _ = execute(store, graph(tmp, steps=6, resume_every=2))
        straight, _ = TinyFamily().read_adapter(str(tmp / "out" / "toy_step000006.safetensors"))
        shutil.rmtree(tmp / "out" / "resume")  # the state of a finished run is not what a cut leaves; start the interrupted pair from nothing
        shutil.move(str(tmp / "out"), str(tmp / "straight"))
        _, first_half, _ = execute(store, graph(tmp, steps=4, resume_every=2))
        saved = sorted(p.name for p in (tmp / "out" / "resume").glob("*.json"))
        for lora in (tmp / "out").glob("toy_step*.safetensors"):
            lora.unlink()  # the cut came before any result was kept; only the resume states are left
        _, second_half, trace = execute(store, graph(tmp, steps=6, resume_every=2))
        resumed, _ = TinyFamily().read_adapter(str(tmp / "out" / "toy_step000006.safetensors"))
        check(whole.ok and first_half.ok and second_half.ok and saved == ["resume_00000002.json", "resume_00000004.json"] and trace.steps() == 2,
              f"a run that stops at step 4 and is started again with more steps trains only the last two ({first_half.error}, {second_half.error}, {saved}, {trace.steps()} steps)")
        check(resumed.keys() == straight.keys() and all(torch.allclose(resumed[k], straight[k], atol=1e-6) for k in straight),
              "the resumed run ends with the same adapter as the run that was never interrupted")
        _, other, _ = execute(store, graph(tmp, steps=6, resume_every=2, lr=2e-3))
        check(not other.ok and "different run" in str(other.error), f"a state of another run (different learning rate) is refused ({other.error})")


if __name__ == "__main__":
    run(full_run)
