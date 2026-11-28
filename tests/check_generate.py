"""Generation nodes: a whole text-to-image graph runs on the CPU with tiny stand-ins for the weights, and what it writes is reproducible.

    python -m tests.check_generate

Covers the Live preview addon of the Sampler (what it shows through Preview Image, where it stays silent, and its refusal to end a run), the two sampling schemes (noise prediction and rectified flow), the adapter list syntax, the preview pictures a page shows, the starter graph (every wire resolves), the run from prompt to PNG files (shape, seeds,
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


def live_preview() -> None:
    """The Live preview addon of the Sampler: it shows through the Preview Image stream, only where something shows it, and never ends a run."""
    from graph.document import AddonSpec

    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        doc = graph(tmp, prompt="a red fox", seed=7, steps=4)
        sampler = next(spec for spec in doc.nodes if spec.class_name == "SamplerNode")
        preview = next(spec.uid for spec in doc.nodes if spec.class_name == "PreviewImageNode")
        check(any(a.kind == "LivePreviewAddon" and a.enabled for a in sampler.addons), "the image starter has the Live preview addon on its sampler")
        validation, outcome, trace = execute(doc, artifacts=tmp / "art")
        check(validation.ok and outcome.ok, f"a sampler with the addon runs ({[i.message for i in validation.errors][:2]} {outcome.error})")
        looks = [f for s, f in trace.frames if s == f"node:{preview}" and f.get("live")]
        check([f["step"] for f in looks] == [2, 4], f"Preview Image gets a look every second step and the last ({[f['step'] for f in looks]})")
        files = [tmp / "art" / f["images"][0]["path"] for f in looks]
        check(all(f.is_file() and f.suffix == ".jpg" for f in files) and len({f.name for f in files}) == 1,
              "one live file under the node's own folder, rewritten for each look")
        check(len({f["images"][0]["v"] for f in looks}) == 2, "each look has its own version, so a page never keeps an older picture")
        check(looks[0]["images"][0]["width"] == 8 and looks[0]["images"][0]["height"] == 8, "a latent2rgb look is as small as the latents")
        final = [f for s, f in trace.frames if s == f"node:{preview}" and not f.get("live") and "images" in f]
        check(len(final) == 1 and final[0]["images"][0]["path"].endswith(".png"), "the finished image then arrives in the same stream")
        check(len(list((tmp / "out").glob("*.png"))) == 1, "and is still written")

        (tmp / "off").mkdir()
        off = graph(tmp / "off", prompt="a red fox", seed=7, steps=4)
        off = replace(off, nodes=tuple(replace(n, addons=tuple(replace(a, enabled=False) for a in n.addons)) for n in off.nodes))
        _, plain, quiet = execute(off, artifacts=tmp / "art2")
        check(plain.ok and not [1 for s, f in quiet.frames if f.get("live")], "with the addon off nothing is shown")
        (tmp / "unwatched").mkdir()
        unwatched = graph(tmp / "unwatched", prompt="a red fox", seed=7, steps=4)
        unwatched = replace(unwatched, nodes=tuple(n for n in unwatched.nodes if n.class_name != "PreviewImageNode"),
                            edges=tuple(e for e in unwatched.edges if e.to_node != preview))
        _, nobody, silent = execute(unwatched, artifacts=tmp / "art3")
        check(nobody.ok and not [1 for s, f in silent.frames if f.get("live")] and not list((tmp / "art3").rglob("*.jpg")),
              "with no Preview Image downstream nothing is decoded or written")
        _, blind, noted = execute(graph(tmp, prompt="a red fox", seed=7, steps=4))
        check(blind.ok and any(s == "warning" and "nothing is shown" in f["message"] for s, f in noted.frames), "a run with nowhere to put pictures says so and goes on")

        vae = replace(sampler, addons=tuple(replace(a, params={**a.params, "method": "vae", "size": 64}) for a in sampler.addons))
        wired = replace(doc, nodes=tuple(vae if n.uid == sampler.uid else n for n in doc.nodes))
        _, bad, notes = execute(wired, artifacts=tmp / "art4")
        check(bad.ok and any(s == "warning" and "needs a VAE" in f["message"] for s, f in notes.frames), "the vae method without a VAE wired says so; the run still finishes")


def live_preview_parts() -> None:
    """The decoders and the cadence of ``LivePreview``, without a graph."""
    from engine.generate.preview import LatentRGB, LivePreview, PreviewUnavailable, VAEPreview

    factors = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    rgb = LatentRGB(factors, (0.0, 0.0, 0.0)).decode(torch.tensor([[[[1.0]], [[-1.0]]]]))
    check(rgb.dtype == torch.uint8 and rgb.shape == (3, 1, 1) and rgb[:, 0, 0].tolist() == [255, 0, 128],
          f"latent2rgb maps [-1, 1] to bytes with a clamp ({rgb[:, 0, 0].tolist()})")
    raises(ValueError, lambda: LatentRGB(factors).decode(torch.zeros(1, 4, 2, 2)), "a projection for another channel count is refused")

    from families.anima.latents import LATENTS as ANIMA
    from families.sdxl.latents import LATENTS as SDXL
    check(all(len(spec.rgb) == spec.channels and all(len(row) == 3 for row in spec.rgb) and len(spec.rgb_bias) == 3 for spec in (SDXL, ANIMA)),
          "each family's colour projection has a row of three per latent channel")

    shown, notes, closed = [], [], []

    class Counting:
        def decode(self, latents):
            return torch.zeros(3, 2, 2, dtype=torch.uint8)

        def close(self):
            closed.append(True)

    live = LivePreview(every=3, choose=lambda family: Counting(), show=lambda p, step, total: shown.append(step), note=notes.append)
    check(not live.active, "nothing is active before the sampler begins")
    live.begin(None)
    for step in (1, 3, 9, 10):
        live(step, 10, torch.zeros(2, 4, 4, 4))
    check(shown == [3, 9, 10], f"a look every N steps and at the end ({shown})")
    live.finish()
    check(closed == [True] and not live.active, "finish drops the decoder")

    def refuse(family):
        raise PreviewUnavailable("no projection")

    refused = LivePreview(every=1, choose=refuse, show=lambda *a: shown.append(-1), note=notes.append)
    refused.begin(None)
    check(not refused.active and notes == ["Live preview: no projection"], "an unavailable preview says why and stays off")
    watching = LivePreview(every=1, choose=lambda family: None, show=lambda *a: shown.append(-1), note=notes.append)
    watching.begin(None)
    check(not watching.active and len(notes) == 1, "with nobody watching it is off without a word")

    codec_loads, released = [], []

    class Codec:
        def decode(self, latents):
            if latents.shape[-1] > 4:
                raise RuntimeError("XPU out of memory")
            return torch.zeros(1, 3, 8, 8, dtype=torch.uint8)

    def load():
        codec_loads.append(True)
        return Codec()

    vae = VAEPreview(size=32, spatial_factor=8, load=load, release=released.append)
    check(vae.shrunk(torch.zeros(1, 4, 128, 96)).shape[-2:] == (4, 3), "latents are shrunk until their decode is about the size asked for")
    check(vae.shrunk(torch.zeros(1, 4, 2, 2)).shape[-2:] == (2, 2), "small latents are left as they are")
    check(not codec_loads, "the VAE is not loaded before the first look")
    vae.decode(torch.zeros(1, 4, 4, 4))
    vae.decode(torch.zeros(1, 4, 4, 4))
    vae.close()
    check(len(codec_loads) == 1 and len(released) == 1, "it is loaded once and released at the end")

    stopped, stop_notes = [], []
    big = LivePreview(every=1, choose=lambda family: VAEPreview(size=1024, spatial_factor=8, load=Codec, release=lambda c: stopped.append(1)),
                      show=lambda *a: stopped.append("shown"), note=stop_notes.append)
    big.begin(None)
    big(1, 5, torch.zeros(1, 4, 16, 16))
    big(2, 5, torch.zeros(1, 4, 16, 16))
    check(not big.active and len(stop_notes) == 1 and "room" in stop_notes[0] and stopped == [1],
          "when the decode does not fit, previews stop for the run, once, and the decoder is released")

    def explode(latents):
        raise RuntimeError("the decoder broke")

    class Broken(Counting):
        decode = staticmethod(explode)

    broken_notes = []
    broken = LivePreview(every=1, choose=lambda family: Broken(), show=lambda *a: None, note=broken_notes.append)
    broken.begin(None)
    broken(1, 5, torch.zeros(1, 4, 4, 4))
    broken(2, 5, torch.zeros(1, 4, 4, 4))
    check(not broken.active and len(broken_notes) == 1 and "the decoder broke" in broken_notes[0], "any other failure also only stops previews, and says why")


def live_file() -> None:
    """The live picture is one JPEG per node, replaced whole, so a page that fetches it never reads half a file."""
    with tempfile.TemporaryDirectory() as raw:
        folder = ArtifactFolder(raw)
        first = folder.put_live("n_aaaaaa", torch.zeros(3, 8, 8, dtype=torch.uint8))
        second = folder.put_live("n_aaaaaa", torch.full((3, 8, 8), 255, dtype=torch.uint8))
        check(first == second and (Path(raw) / first).is_file() and not list(Path(raw).rglob("*.part.*")), "the same file is replaced and no temporary one is left")


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


def preview_names() -> None:
    """A picture shown on a page never takes the address of a different one: the same prompt and seed from another model, adapter or setting is another file."""
    from engine.generate.recipe import Image, Recipe
    with tempfile.TemporaryDirectory() as raw:
        folder = ArtifactFolder(raw)

        def name(recipe: Recipe, **meta) -> str:
            return folder.put("n_aaaaaa", [Image(recipe, torch.zeros(3, 8, 8, dtype=torch.uint8), {"recipes": [recipe.seed], **meta})])[0]

        base = Recipe(prompt="p", seed=1)
        names = {name(base, model="a.safetensors", adapters=[]), name(base, model="b.safetensors", adapters=[]), name(replace(base, cfg=3.0), model="a.safetensors", adapters=[]),
                 name(base, model="a.safetensors", adapters=[{"file": "x", "strength": 1}])}
        check(len(names) == 4, f"model, setting and adapter each give another file ({len(names)} of 4)")
        check(name(base, model="a.safetensors", adapters=[]) in names, "and the same picture again keeps its own name")


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
    live_preview()
    live_preview_parts()
    live_file()
    preview_housekeeping()
    preview_names()
    refusals()
    print("\ndone")
