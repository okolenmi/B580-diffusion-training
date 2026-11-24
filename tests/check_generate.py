"""Generation nodes: a whole text-to-image graph runs on the CPU with tiny stand-ins for the weights, and what it writes is reproducible.

    python -m tests.check_generate

Covers the live sampling preview (a node wired into the sampler, and its refusal to end a run), the two sampling schemes (noise prediction and rectified flow), the adapter list syntax, the preview pictures a page shows, the starter graph (every wire resolves), the run from prompt to PNG files (shape, seeds,
the recipe inside each file, stopping), and that the same recipe gives the same pixels alone or in a batch.
"""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

import starters
from engine.devices import Accelerator
from engine.generate.artifacts import ArtifactFolder
from engine.generate.recipe import ImageStore
from engine.runtime.factory import make_runtime
from dataclasses import replace

import torch

from engine.generate.preview import SamplingPreview
from graph import GraphRunner, NodeRegistry
from graph.node import Environment
from nodes.model.adapter_stack import adapter_rows, parse_stack
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
    services = {"families": lambda _id: FAMILY}
    if artifacts:
        services["artifacts"] = ArtifactFolder(artifacts)
    env = Environment(services=services)
    runner = GraphRunner(REGISTRY, reporter=trace, env=env, runtime_factory=make_runtime, observer=NodeTrace(doc, Accelerator("cpu"), None))
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
    rows = [{"file": "a.safetensors", "strength": 0.8, "on": True}, {"file": "b.safetensors", "strength": 1, "on": False}, {"file": "c.safetensors", "strength": "0.3"}]
    check(adapter_rows(rows) == [("a.safetensors", 0.8), ("c.safetensors", 0.3)], "the list control's rows give the adapters that are on, in order")
    check(adapter_rows(None) == [] and adapter_rows("a.safetensors=2") == [("a.safetensors", 2.0)], "no rows is no adapters; a graph saved with the text form still reads")
    raises(ValueError, lambda: adapter_rows([{"file": "a", "strength": "loud"}]), "a strength that is not a number is refused", "not a strength")
    raises(ValueError, lambda: adapter_rows([{"strength": 1}]), "a row without a file is refused", "no file")


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


def sampling_preview() -> None:
    from graph.document import Edge, NodeSpec

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        doc = graph(tmp, prompt="a red fox", seed=7, steps=4)
        sampler = next(spec.uid for spec in doc.nodes if spec.class_name == "SamplerNode")
        source = next(spec.uid for spec in doc.nodes if spec.title == "model")
        watching = replace(doc, nodes=doc.nodes + (NodeSpec("n_zzzzzz", "SamplingPreviewNode", {"every": 2, "size": 64}),),
                           edges=doc.edges + (Edge(source, "vae", "n_zzzzzz", "vae"), Edge("n_zzzzzz", "preview", sampler, "preview")))
        validation, outcome, trace = execute(watching, artifacts=tmp / "art")
        check(validation.ok and outcome.ok, f"a sampler with a Sampling Preview wired in runs ({[i.message for i in validation.errors][:2]} {outcome.error})")
        looks = [f for s, f in trace.frames if s == "node:n_zzzzzz" and "images" in f]
        check([f["step"] for f in looks] == [2, 4], f"it shows every second step and the last ({[f['step'] for f in looks]})")
        files = [tmp / "art" / f["images"][0]["path"] for f in looks]
        check(all(f.is_file() for f in files) and len({f.name for f in files}) == 2, "each look is its own picture file under the node's folder")
        check(all(f["images"][0]["width"] <= 64 and f["images"][0]["height"] <= 64 for f in looks), "a look is no larger than the size asked for")
        check(len(list((tmp / "out").glob("*.png"))) == 1, "the finished image is still written")
        (tmp / "plain").mkdir()
        _, plain, quiet = execute(graph(tmp / "plain", prompt="a red fox", seed=7, steps=4), artifacts=tmp / "art2")
        check(plain.ok and not [1 for s, f in quiet.frames if s == "node:n_zzzzzz"], "without the node nothing changes")

    shown, notes, closed = [], [], []
    def decode(latents):
        if latents.shape[-1] > 4:
            raise RuntimeError("XPU out of memory")
        return torch.zeros(1, 3, 8, 8, dtype=torch.uint8)
    preview = SamplingPreview(every=3, size=32, spatial_factor=8, decode=decode, show=lambda p, step, total: shown.append(step),
                              note=notes.append, close=lambda: closed.append(True))
    preview(1, 10, torch.zeros(2, 4, 4, 4))
    preview(3, 10, torch.zeros(2, 4, 4, 4))
    preview(10, 10, torch.zeros(2, 4, 4, 4))
    check(shown == [3, 10], f"a preview is made every N steps and at the end ({shown})")
    check(preview.shrunk(torch.zeros(1, 4, 128, 96)).shape[-2:] == (4, 3), "latents are shrunk until their decode is about the size asked for")
    check(preview.shrunk(torch.zeros(1, 4, 2, 2)).shape[-2:] == (2, 2), "small latents are left as they are")
    big = SamplingPreview(every=1, size=1024, spatial_factor=8, decode=decode, show=lambda p, step, total: shown.append(step),
                          note=notes.append, close=lambda: closed.append(True))
    big(1, 5, torch.zeros(1, 4, 16, 16))
    big(2, 5, torch.zeros(1, 4, 16, 16))
    check(big.stopped and len(notes) == 1 and closed == [True] and shown == [3, 10], "when the decode does not fit, previews stop for the run, once, and the decoder is dropped")


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


def anima_starter() -> None:
    doc = starters.anima_image(model="dit.safetensors", output="out", text_encoder="q.safetensors", vae="v.safetensors")
    classes = {spec.uid: (spec.class_name, spec.params) for spec in doc.nodes}
    check(all(cls in REGISTRY.load()[0] for cls, _ in classes.values()), "every node of the Anima image graph exists")
    scopes = [p.get("scope") for c, p in classes.values() if c == "LoadModelNode"]
    check(scopes == ["diffusion_models", "text_encoders", "vae"], f"the DiT, Qwen3 and VAE are three Load Model nodes ({scopes})")
    wires = {(e.from_node, e.from_port, e.to_node, e.to_port) for e in doc.edges}
    check(("n_bbbbbb", "qwen3", "n_eeeeee", "encoders") in wires and ("n_cccccc", "vae", "n_gggggg", "vae") in wires, "Qwen3 feeds the prompt, the VAE file feeds the decoder")
    sampler_params = next(p for c, p in classes.values() if c == "SamplerNode")
    check((sampler_params["sampler"], sampler_params["scheduler"]) == ("euler", "simple"), "it samples with Euler on the simple schedule")
    raises(ValueError, lambda: starters.anima_image(model="dit.safetensors", output="out"), "the starter needs the two other files", "text encoder")


def flow_scheme() -> None:
    import torch
    from engine.generate.recipe import Recipe
    from engine.generate.scheme import EpsScheme, FlowScheme, scheme_for
    from families.registry import get_family

    scheme = FlowScheme(shift=3.0)
    sigmas = scheme.sigmas(2)
    check([round(float(v), 4) for v in sigmas] == [1.0, 0.75, 0.0], f"flow positions are the shifted line from 1 to 0 ({sigmas.tolist()})")
    many = scheme.sigmas(30)
    check(bool((many[1:] < many[:-1]).all()) and float(many[0]) == 1.0 and float(many[-1]) == 0.0, "30 flow positions fall monotonically from 1 to 0")
    check(isinstance(scheme_for(get_family("sdxl")), EpsScheme), "SDXL declares the noise-prediction scheme")
    anima = scheme_for(get_family("anima"))
    check(isinstance(anima, FlowScheme) and anima.shift == 3.0, "Anima declares the flow scheme with its objective's shift")

    for bad, why in ((Recipe("x", sampler="dpmpp_2m", scheduler="simple"), "dpmpp_2m"), (Recipe("x", sampler="euler", scheduler="karras"), "karras"),
                     (Recipe("x", sampler="euler", scheduler="simple", eps_scale=1.05), "epsilon")):
        raises(ValueError, lambda r=bad: scheme.check(r), f"a flow model refuses {why}", why)
    scheme.check(Recipe("x", sampler="euler", scheduler="simple"))

    # A network that knows the clean sample predicts the velocity (x - x0) / s exactly; Euler then lands on x0 whatever the steps and guidance.
    x0 = torch.randn(1, 16, 4, 4)
    for steps, cfg in ((1, 1.0), (7, 4.0)):
        recipes = (Recipe("x", seed=5, steps=steps, cfg=cfg, sampler="euler", scheduler="simple", width=32, height=32),)
        out = scheme.sample(recipes, lambda r: (1, 16, 4, 4), lambda x, t, positive: (x - x0) / t.view(-1, 1, 1, 1), cfg, "cpu")
        check(bool(torch.allclose(out, x0, atol=1e-4)), f"flow Euler recovers the clean sample in {steps} step(s) at guidance {cfg}")
    check(not torch.equal(scheme.sample(recipes, lambda r: (1, 16, 4, 4), lambda x, t, p: x * 0, 1.0, "cpu"),
                          scheme.sample((Recipe("x", seed=6, steps=7, sampler="euler", scheduler="simple"),), lambda r: (1, 16, 4, 4), lambda x, t, p: x * 0, 1.0, "cpu")),
          "another seed starts from other noise")


if __name__ == "__main__":
    anima_starter()
    flow_scheme()
    adapter_list()
    run_graph()
    sampling_preview()
    preview_housekeeping()
    refusals()
    print("\ndone")
