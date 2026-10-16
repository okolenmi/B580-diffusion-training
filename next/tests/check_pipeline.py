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

from ..data.store import DatasetStore
from ..engine.runtime.factory import make_runtime
from ..graph import Edge, GraphDoc, GraphRunner, NodeRegistry, NodeSpec
from ..graph.node import Environment
from ..nodes.conditioning import conditioner
from ..nodes.dataset import batching, latents
from ..nodes.model import inject_adapter
from .check_data import make_images
from .pipeline_nodes import TinyFamily
from .support import check, run

REGISTRY = NodeRegistry(packages=("next.nodes", "next.tests.pipeline_nodes"))
NODES = ["TinyModelSource", "DatasetLibraryNode", "LatentCacheNode", "CaptionsNode", "TagSwapNode", "BucketerNode", "NoiserNode",
         "BatcherNode", "ConditionerNode", "InjectAdapterNode", "AdamWNode", "ConstantLRNode", "LossWeightingNode",
         "DiffusionTrainerNode", "EveryNStepsNode", "SaveAdapterNode"]


def uid(index: int) -> str:
    return "n_" + "abcdefghijklmnop"[index] * 6


def graph(tmp: Path, steps: int, params: dict[str, dict] | None = None) -> GraphDoc:
    params = params or {}
    base = {
        0: {"file": str(tmp / "model.safetensors")}, 1: {"dataset": "toy"}, 2: {"dataset": "toy", "base": 64, "mode": "fit", "batch_size": 2},
        5: {"multiple": 1, "keep_incomplete": True}, 7: {"batch_size": 2, "seed": 3}, 9: {"rank": 4, "alpha": 4.0}, 10: {"lr": 1e-3},
        11: {"lr": 1e-3}, 12: {"kind": "min_snr"}, 13: {"steps": steps}, 14: {"every": 2},
        15: {"directory": str(tmp / "out"), "name": "toy"},
    }
    for key, value in (params or {}).items():
        base.setdefault(int(key), {}).update(value)
    specs = tuple(NodeSpec(uid(i), cls, base.get(i, {})) for i, cls in enumerate(NODES))
    wires = [(0, "vae", 2, "vae"), (1, "images", 2, "images"), (1, "images", 3, "images"), (3, "prompts", 4, "prompts"),
             (2, "latents", 5, "latents"), (0, "denoiser", 6, "denoiser"), (2, "latents", 7, "latents"), (4, "prompts", 7, "prompts"),
             (6, "noiser", 7, "noiser"), (5, "buckets", 7, "buckets"), (0, "clip_l", 8, "encoders"), (0, "clip_g", 8, "encoders"),
             (4, "prompts", 8, "prompts"), (0, "denoiser", 9, "denoiser"), (9, "model", 10, "model"), (7, "batches", 13, "batches"),
             (9, "model", 13, "model"), (8, "conditioner", 13, "conditioner"), (10, "optimizer", 13, "optimizer"),
             (11, "schedule", 13, "schedule"), (12, "weighting", 13, "weighting"), (9, "model", 15, "model")]
    events = [(7, "prompts_changed", 8, "prompts_changed"), (13, "step_finished", 14, "step"), (14, "due", 15, "checkpoint"),
              (13, "ended", 15, "checkpoint")]
    edge = lambda w: Edge(uid(w[0]), w[1], uid(w[2]), w[3])  # noqa: E731
    return GraphDoc(specs, tuple(map(edge, wires)), tuple(map(edge, events)),
                    {"device.default": "cpu", "precision.denoiser": "fp32", "precision.encoders": "fp32", "precision.autocast": "off"})


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
