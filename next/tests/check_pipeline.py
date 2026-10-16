"""Acceptance: a whole training graph runs through the graph runtime on the CPU, from dataset to saved adapter.

    python -m next.tests.check_pipeline

The graph is the one a person builds in the editor (dataset chain, conditioner, adapter, optimizer, schedule,
trainer, checkpoint trigger, saver) with only the model weights replaced by tiny stand-ins. It proves the nodes
agree with each other and with the runtime: ports, event wiring and order, stopping, and the files that come out.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from .. import starters
from ..data.store import DatasetStore
from ..engine.runtime.factory import make_runtime
from ..graph import GraphDoc, GraphRunner, NodeRegistry
from ..graph.node import Environment
from ..nodes.conditioning import conditioner
from ..nodes.dataset import batching, latents
from ..nodes.model import inject_adapter
from .check_data import make_images
from .pipeline_nodes import TinyFamily
from .support import check, run

REGISTRY = NodeRegistry(packages=("next.nodes", "next.tests.pipeline_nodes"))
SETTINGS = {"device.default": "cpu", "precision.denoiser": "fp32", "precision.encoders": "fp32", "precision.autocast": "off"}


def graph(tmp: Path, steps: int) -> GraphDoc:
    return starters.sdxl_lora(model="", dataset="toy", output=str(tmp / "out"), name="toy", resolution=64, batch_size=2, steps=steps,
                              lr=1e-3, rank=4, save_every=2, settings=SETTINGS,
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
    runner = GraphRunner(REGISTRY, reporter=trace, env=Environment(services={"datasets": store}), runtime_factory=make_runtime)
    with ExitStack() as stack:
        for module in (conditioner, inject_adapter, latents, batching):
            stack.enter_context(mock.patch.object(module, "get_family", lambda _id: TinyFamily()))
        validation = runner.validate(doc)
        outcome = runner.run(doc, cancel)
    return validation, outcome, trace


def full_run():
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        folder = tmp / "images"
        folder.mkdir()
        make_images(folder)
        store = DatasetStore(tmp / "datasets")
        library = store.library("toy", create=True)
        library.ingest(folder)
        library.close()

        validation, outcome, trace = execute(store, graph(tmp, steps=4))
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


if __name__ == "__main__":
    run(full_run)
