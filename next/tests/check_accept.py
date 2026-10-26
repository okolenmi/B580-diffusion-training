"""Acceptance lab: regions, the block recorder, the quantization probe (a planted fault is found where it was planted), fixed denoiser inputs.

    python -m next.tests.check_accept
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from ..accept import checks as C
from ..accept.acceptance import Acceptance
from ..accept.artifacts import RunArtifacts
from ..accept.checks import Status
from ..accept.fixtures import DenoiserInputs
from ..accept.quantization import QuantizationProbe, relative_error
from ..accept.recorder import BlockRecorder
from ..accept.regions import Region, Regions
from ..families.general.ldm_unet.unet import UNetModel
from ..families.general.quantize.backends import TorchBackend
from ..families.general.quantize.int8 import Int8Quantizer
from ..families.general.quantize.nf4 import Nf4Quantizer
from ..families.general.quantize.nf4_backends import TorchNf4Backend
from ..families.sdxl.denoiser import SDXLDenoiser
from ..perf.matrix import Outcome, Scenario, SubprocessRunner
from .check_sdxl import TINY_UNET
from ..accept.suites import SuiteBook
from ..accept.visual import ImageSheet, PaddingSheet, to_pil
from ..accept.verdict import Judge, write_verdict
from .check_perf import synthetic
from .support import check, run


def tiny_denoiser() -> SDXLDenoiser:
    torch.manual_seed(0)
    return SDXLDenoiser(UNetModel(**TINY_UNET), dtype=torch.float32)


def tiny_inputs() -> DenoiserInputs:
    return DenoiserInputs(resolution=128, timesteps=[100, 700], device="cpu", dtype=torch.float32, context_dim=48, adm_dim=40, tokens=5)


def regions():
    r = Regions()
    check(r.of("input_blocks.4.1.transformer_blocks.0.attn1.to_q") == Region("down", 4) and r.of("middle_block.1.proj_in") == Region("mid", None)
          and r.of("output_blocks.2.0.in_layers.2") == Region("up", 2) and r.of("time_embed.0") == Region("embed", None) and r.of("out.2") == Region("out", None),
          "paths map to their regions")
    check(Regions(coarse=True).of("input_blocks.4.0.x").name == "down" and r.of("input_blocks.4.0.x").name == "down.4", "coarse regions drop the block index")
    names = [x.name for x in r.order({Region("up", 1), Region("down", 3), Region("mid", None), Region("down", 1), Region("embed", None)})]
    check(names == ["embed", "down.1", "down.3", "mid", "up.1"], f"regions order from the input to the output ({names})")


def recorder():
    den, inputs = tiny_denoiser(), tiny_inputs()
    with BlockRecorder(den.module()) as rec:
        out, grad = inputs.run(den)
    rec = rec.recording
    check(rec.outputs and set(rec.grads) <= set(rec.outputs) and len(rec.grads) == len(rec.outputs), f"every block's output and incoming gradient are kept ({len(rec.outputs)} blocks)")
    check(all(t.dtype == torch.float32 and t.device.type == "cpu" for t in rec.outputs.values()), "recordings live on the host in fp32")
    check(not any(m._forward_hooks for m in den.module().modules()), "the hooks are gone after the block")
    again = tiny_inputs().run(den)
    check(torch.equal(again[0], out) and torch.equal(again[1], grad), "the fixed inputs give the same output and gradient every time")
    with BlockRecorder(den.module(), limit_mb=0.01) as small:
        inputs.run(den)
    check(small.recording.dropped > 0 and small.recording.megabytes <= 0.01 + 1e-9, "the memory limit drops blocks and counts them")


def nf4_probe():
    den, inputs = tiny_denoiser(), tiny_inputs()
    quantizer = Nf4Quantizer(TorchNf4Backend(), min_params=0)
    probe = QuantizationProbe()
    clean = probe.run(den.module(), quantizer, lambda _: inputs.run(den), precision="nf4")
    check(clean.report.layers > 0 and clean.weights and all(0 < e < 0.2 for e in clean.weights.values()), "every converted layer has a plausible weight error")
    check(0 < clean.final["output"] < 0.5 and 0 < clean.final["input_grad"] < 1.0, f"the whole network moved a little ({clean.final})")
    groups = {r.region.split(".")[0] for r in clean.regions}
    check({"down", "mid", "up"} <= groups, f"errors are reported by region ({sorted(r.region for r in clean.regions)[:6]} ...)")
    text = clean.markdown()
    check("## By region" in text and "weight error" in text and clean.to_json()["regions"], "the report has the region table and the worst layers")

    fault = "input_blocks.5.1.transformer_blocks.0.ff.net.2"

    def plant(model: nn.Module):
        report = quantizer.run(model)
        layer = dict(model.named_modules())[fault]
        layer.weight_absmax.mul_(8.0)
        return report

    den2 = tiny_denoiser()
    planted = probe.run(den2.module(), Nf4Quantizer(TorchNf4Backend(), min_params=0), lambda _: inputs.run(den2), precision="nf4", convert=plant)
    before = [p for p in clean.modules if p.startswith(("input_blocks.1", "input_blocks.2", "input_blocks.3", "input_blocks.4"))]
    check(before and all(abs(planted.modules[p]["output"] - clean.modules[p]["output"]) < 1e-7 for p in before), "blocks upstream of the fault are unaffected in the forward pass")
    hit = "input_blocks.5.1"
    check(planted.modules[hit]["output"] > 3 * clean.modules[hit]["output"], "the block holding the fault shows a much larger output error")
    check(planted.final["output"] > clean.final["output"], "and so does the whole network")


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b = nn.Linear(256, 256, bias=False), nn.Linear(256, 256, bias=False)

    def forward(self, x):
        return x + self.b(torch.relu(self.a(x)))


def int8_probe():
    torch.manual_seed(0)
    model = nn.Sequential(Block(), Block())
    x = torch.randn(8, 256)
    dy = torch.randn(8, 256)

    def run_model(m):
        inp = x.clone().requires_grad_(True)
        out = m(inp)
        out.backward(dy)
        return out, inp.grad

    result = QuantizationProbe(select=lambda m: isinstance(m, Block)).run(model, Int8Quantizer(TorchBackend(), min_params=0), run_model, precision="int8")
    check(result.report.layers == 4 and set(result.modules) == {"0", "1"} and all(0 < e < 0.05 for e in result.weights.values()), "int8 layers convert and are measured")
    check(0 < result.final["output"] < 0.1 and relative_error(torch.ones(3), torch.ones(3)) == 0 and relative_error(torch.zeros(2), torch.zeros(2)) != relative_error(torch.zeros(2), torch.zeros(2)),
          "relative error is zero for equal tensors and undefined against a zero reference")


def log_text(steps, *, quant=None, graphs=(), budget=None, ok=True, extra=()) -> str:
    """A console log in the real format. ``steps`` are (loss, ms, peak_mb, shape-or-None, new) tuples."""
    lines = []
    if budget:
        lines.append(f"memory budget: {budget} MB of the card's 12288 MB, 800 MB in use now, margin 512 MB; allocator config: default")
    if quant:
        lines.append(quant)
    lines += list(extra)
    for i, (loss, ms, peak, shape, new) in enumerate(steps, 1):
        where = f"  [{shape[0]}x{shape[1]}x{shape[2]}{' new' if new else ''}]" if shape else ""
        lines.append(f"step {i}/{len(steps)}  loss {loss:.4f}  {ms} ms  epoch 1{where}  graphs 1 (3 replays)  device {peak + 600} MB (live {peak - 500}, step peak {peak}, cached 900)")
    lines += list(graphs)
    lines.append("run: ok" if ok else "run: failed: ValueError: boom")
    return "\n".join(lines) + "\n"


def make_run(root: Path, scenario: str, steps, **kw) -> Path:
    folder = root / Scenario(scenario).slug
    folder.mkdir(parents=True)
    (folder / "run.log").write_text(log_text(steps, **kw))
    return folder


def curve(loss=0.1, ms=1000, peak=8000, count=14, new_first=True):
    return [(loss, ms * 3 if i == 0 and new_first else ms, peak, (3, 64, 64), i == 0 and new_first) for i in range(count)]


def artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        folder = make_run(Path(tmp), "x", curve(), quant="int8 base weights (xpu_kernels): 652 layers, 2059 MB saved; 10 left as they were (small)", budget=10240,
                          graphs=["[graph] shape (3, 64, 64): replay 900 ms against 1000 ms eager", "[graph] shape (2, 64, 96): replay 1100 ms against 1000 ms eager; the graph does not pay and is dropped"])
        art = RunArtifacts.read(folder)
        first, second = art.steps[0], art.steps[1]
        check(len(art.steps) == 14 and first.shape == (3, 64, 64) and first.new_shape and not second.new_shape and second.loss == 0.1 and second.ms == 1000, "step lines become records")
        check(second.peak_mb == 8000 and second.device_mb == 8600 and second.graphs == 1 and second.replays == 3, "memory and graph fields are read")
        check(art.quantization == {"label": "int8", "backend": "xpu_kernels", "layers": 652, "saved_mb": 2059} and art.memory_limit_mb == 10240 and art.finished, "conversion, budget and the final line")
        check([(d.shape, d.dropped) for d in art.graph_decisions] == [("(3, 64, 64)", False), ("(2, 64, 96)", True)], "graph decisions are read, kept and dropped")
        check(len(art.settled_steps()) == 13 and art.files("*.log"), "settled steps leave out the first sighting of a shape")
        failed = RunArtifacts.read(make_run(Path(tmp), "y", curve(count=2), ok=False))
        check(not failed.finished and "boom" in failed.failed_with, "a failed run is recognised")
        check(RunArtifacts.read(Path(tmp) / "nothing").lines == [], "a missing folder reads as an empty run")


def checks():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = RunArtifacts.read(make_run(root, "base", curve(loss=0.100, ms=1000, peak=8000)))
        fast = RunArtifacts.read(make_run(root, "fast", curve(loss=0.105, ms=900, peak=6000)))
        slow = RunArtifacts.read(make_run(root, "slow", curve(loss=0.300, ms=1500, peak=9000)))
        broken = RunArtifacts.read(make_run(root, "broken", curve(count=3), ok=False))
        nan = RunArtifacts.read(make_run(root, "nan", [(float("nan"), 1000, 8000, None, False)]))
        status = lambda c, *runs: c.evaluate(list(runs)).status  # noqa: E731
        check(status(C.RunFinished("f", ("a",)), base) is Status.PASS and status(C.RunFinished("f", ("a",)), broken) is Status.FAIL and "boom" in C.RunFinished("f", ("a",)).evaluate([broken]).message,
              "a finished run passes, a failed one fails and says why")
        check(status(C.LinesPresent("l", ("a",), r"^step 1/"), base) is Status.PASS and status(C.LinesPresent("l", ("a",), r"nothing here"), base) is Status.FAIL
              and status(C.LinesAbsent("l", ("a",), r"boom"), base) is Status.PASS and status(C.LinesAbsent("l", ("a",), r"^run: ok"), base) is Status.FAIL, "line checks count matches")
        check(status(C.LossFinite("l", ("a",)), base) is Status.PASS and status(C.LossFinite("l", ("a",)), nan) is Status.FAIL, "a NaN loss fails")
        check(status(C.PeakBelow("p", ("a",), 8500), base) is Status.PASS and status(C.PeakBelow("p", ("a",), 7000), base) is Status.FAIL
              and status(C.PeakAtLeast("p", ("a",), 8000, "peak_mb"), base) is Status.PASS and status(C.PeakAtLeast("p", ("a",), 9000, "peak_mb"), base) is Status.FAIL, "peak limits both ways")
        check(status(C.PeakLower("p", ("a", "b"), 1500), fast, base) is Status.PASS and status(C.PeakLower("p", ("a", "b"), 3000), fast, base) is Status.FAIL, "a peak lower by at least a margin")
        check(status(C.StepTimeWithin("t", ("a", "b"), 1.0), fast, base) is Status.PASS and status(C.StepTimeWithin("t", ("a", "b"), 1.15), slow, base) is Status.FAIL, "step time ratio")
        check(status(C.StepTimeBelow("t", ("a",), 1200), base) is Status.PASS and status(C.StepTimeBelow("t", ("a",), 800), base) is Status.FAIL, "step time limit")
        check(status(C.StepTimeBelow("t", ("a",), 1200), nan) is Status.PASS, "a single-step log still has a time")
        check(status(C.LossCurveClose("c", ("a", "b"), 0.10), fast, base) is Status.PASS and status(C.LossCurveClose("c", ("a", "b"), 0.10), slow, base) is Status.FAIL, "loss curves within a band")
        check(status(C.LossCurveClose("c", ("a", "b"), 0.0, 1000), base, base) is Status.PASS, "identical curves pass a zero tolerance")
        graphed = RunArtifacts.read(make_run(root, "graphed", curve(), graphs=["[graph] shape (3, 64, 64): replay 900 ms against 1000 ms eager"]))
        dropped = RunArtifacts.read(make_run(root, "dropped", curve(), graphs=["[graph] shape (3, 64, 64): replay 1100 ms against 1000 ms eager; the graph does not pay and is dropped"]))
        check(status(C.GraphDecided("g", ("a",), "kept"), graphed) is Status.PASS and status(C.GraphDecided("g", ("a",), "kept"), dropped) is Status.FAIL
              and status(C.GraphDecided("g", ("a",), "dropped"), dropped) is Status.PASS and status(C.GraphDecided("g", ("a",), "any"), base) is Status.FAIL, "graph decisions against what was expected")
        shown = C.NumbersShown("n", ("a", "b"), "peak_mb", "Same?").evaluate([base, fast])
        check(shown.status is Status.LOOK and "a 8000 MB, b 6000 MB" in shown.message, "a diagnostic shows both numbers and asks for a look")
        for name, a_value in (("same", 1.0), ("different", 2.0)):
            for tag in ("a", "b"):
                folder = root / f"{name}_{tag}" / "out"
                folder.mkdir(parents=True)
                save_file({"w": torch.ones(2, 2) * (a_value if tag == "b" else 1.0)}, str(folder / "lora.safetensors"))
        same = [RunArtifacts.read(root / "same_a"), RunArtifacts.read(root / "same_b")]
        different = [RunArtifacts.read(root / "different_a"), RunArtifacts.read(root / "different_b")]
        ident = C.AdaptersIdentical("i", ("a", "b"))
        check(ident.evaluate(same).status is Status.PASS and ident.evaluate(different).status is Status.FAIL and ident.evaluate([base, base]).status is Status.SKIP, "adapter files compare exactly")
        look = C.LookAt("v", ("a",), instruction="Do the strips look right?", patterns=("out/*.safetensors",)).evaluate([same[0]])
        check(look.status is Status.LOOK and "out/lora.safetensors" in look.message and C.LookAt("v", ("a",), patterns=("*.png",)).evaluate([base]).status is Status.SKIP,
              "a visual check names the files to open, or is skipped when there are none")
        report = RunArtifacts.read(make_run(root, "withreport", curve()))
        report.report = {"findings": [{"title": "The step is launch bound"}]}
        check(C.ReportFinding("r", ("a",), "launch bound").evaluate([report]).status is Status.FAIL and C.ReportFinding("r", ("a",), "launch bound", absent=False).evaluate([report]).status is Status.PASS
              and C.ReportFinding("r", ("a",), "x").evaluate([base]).status is Status.SKIP, "findings of the perf report can be required or forbidden")


def suites_and_verdict():
    book = SuiteBook()
    suites = book.all()
    names = [s.slug for suite in suites.values() for s in suite.scenarios if s.name in ("deterministic a", "limit 9000 int8", "nf4 eager")]
    check(len(names) == 3, "the suites build")
    for suite in suites.values():
        known = set(suite.scenario_names())
        check(all(set(c.scenarios) <= known for c in suite.checks), f"{suite.name}: every check names scenarios of its suite")
    by_name: dict[str, tuple] = {}
    for suite in suites.values():
        for s in suite.scenarios:
            check(by_name.setdefault(s.name, s.args) == s.args, f"scenario {s.name!r} means the same in every suite") if s.name in by_name else by_name.setdefault(s.name, s.args)
    slugs = {s.slug for suite in suites.values() for s in suite.scenarios}
    check(len(slugs) == len({s.name for suite in suites.values() for s in suite.scenarios}), "scenario folders do not collide")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "matrix"
        for scenario, peak, ms in (("native eager", 8000, 1000), ("int8 eager", 6000, 1300), ("native graph", 8400, 900), ("int8 graph", 6400, 950)):
            make_run(root, scenario, curve(peak=peak, ms=ms), graphs=["[graph] shape (3, 64, 64): replay 900 ms against 1000 ms eager"],
                     quant="int8 base weights (xpu_kernels): 652 layers, 2059 MB saved" if scenario.startswith("int8") else None)
        verdict = Judge(root).judge(suites["speed"])
        results = {r.name: r for r in verdict.results}
        check(results["int8 holds less memory than native"].status is Status.PASS and results["int8 graph within 15% of native graph"].status is Status.PASS,
              "int8 saves memory and its graph step is within 15%")
        check(results["dense gradient path no slower than int8"].status is Status.SKIP and "no run for" in results["dense gradient path no slower than int8"].message
              and "int8 dx-int8 graph" in verdict.missing, "checks whose run is missing are skipped and the missing runs are listed")
        check(verdict.status is Status.PASS and verdict.count(Status.FAIL) == 0, "a suite whose evaluated checks all pass passes, skipped checks aside")
        suites["speed"].checks.append(C.LossFinite("explode", ("native eager",)))

        class Exploding(C.Check):
            def evaluate(self, runs):
                raise RuntimeError("bug in the check")
        suites["speed"].checks.append(Exploding("broken check", ("native eager",)))
        again = Judge(root).judge(suites["speed"])
        check(any(r.name == "broken check" and r.status is Status.FAIL and "bug in the check" in r.message for r in again.results) and again.status is Status.FAIL,
              "a check that raises is a failed result and fails the suite")
        path = write_verdict([verdict, again], Path(tmp) / "out")
        text = path.read_text()
        check("# Acceptance verdict" in text and "| Speed of native" in text and "FAIL" in text and (path.parent / "verdict.json").exists(), "the verdict file has the summary and the details")


class FakeRunner:
    def __init__(self, command):
        self.command, self.ran = command, []

    def run(self, scenario, folder):
        self.ran.append(scenario.name)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.log").write_text(log_text(curve(), quant="nf4 base weights (torch): 600 layers, 3000 MB saved" if "nf4" in scenario.name else None))
        (folder / "report.json").write_text(json.dumps(synthetic(1000, 700, 50, 10.0)))
        return Outcome(scenario, True, 0.1, folder, folder / "run.log")


def run_suites():
    book = SuiteBook()
    suites = [book.nf4(), book.allocator()]
    with tempfile.TemporaryDirectory() as tmp:
        runners = []
        acceptance = Acceptance(suites, Path(tmp), ["--file", "m.safetensors"], steps=20, runner_for=lambda c: runners.append(FakeRunner(c)) or runners[-1], say=lambda text: None)
        check("--steps" in acceptance.command and acceptance.command[-1] == "20" and acceptance.command[3:5] == ["sdxl-lora", "--file"], "the command is next.run sdxl-lora with the user's arguments and the step count")
        names = [s.name for s in acceptance.scenarios()]
        check(len(names) == len(set(names)) and "native eager" in names and "nf4 graph" in names, "scenarios of the chosen suites are run once each")
        path = acceptance.run()
        check(sorted(runners[0].ran) == sorted(names) and path.exists() and "Acceptance verdict" in path.read_text(), "every scenario runs and a verdict is written")
        check("nf4 loss finite" in path.read_text() or "nf4 eager loss finite" in path.read_text(), "the verdict lists the checks")
        before = len(runners[0].ran)
        acceptance.run()
        check(len(runners[0].ran) == before, "rerunning keeps the scenarios that already finished")
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "f"
        runner = SubprocessRunner([sys.executable, "-c", "import sys; print(sys.argv[1:])"])
        outcome = runner.run(Scenario("probe", ("--output", "{folder}/out")), folder)
        log = (folder / "run.log").read_text()
        check(outcome.ok and f"{folder}/out" in log and "{folder}" not in log, "{folder} in a scenario's arguments becomes that scenario's folder")


def visual_aids():
    from PIL import Image
    sheet = ImageSheet((64, 48), columns=2, title="t")
    for i in range(3):
        sheet.add(torch.rand(3, 30 + i, 40), f"image {i}")
    sheet.add(Image.new("RGB", (200, 100)), "wide")
    rendered = sheet.render()
    check(len(sheet.items) == 4 and rendered.width == 2 * (64 + 6) + 6 and rendered.height > 2 * 48, f"the grid has two columns and two rows ({rendered.size})")
    check(to_pil(torch.rand(1, 8, 8)).mode == "RGB" and to_pil(torch.ones(3, 4, 4)).getpixel((0, 0)) == (255, 255, 255), "tensors and single channels become RGB images")
    with tempfile.TemporaryDirectory() as tmp:
        check(sheet.save(Path(tmp) / "a" / "sheet.png").exists(), "the sheet is saved, creating folders")
    padding = PaddingSheet((64, 80), sigma=0.6, seed=3)
    images = [Image.fromarray((torch.rand(40, 60, 3) * 255).byte().numpy()) for _ in range(2)]
    result = padding.sheet(images)
    check(len(result.items) == 6 and [c.split(": ")[1].split(" (")[0] for _, c in result.items[:3]] == ["clean, on the canvas", "zero padding", "mirror padding"], "three pictures per sample")
    import random
    clean, (top, left, h, w) = padding.place(images[0], random.Random(3))
    check(clean.shape == (3, 64, 80) and h <= 64 and w <= 80 and float(clean[:, :top].abs().sum()) == 0.0 and float(clean[:, top:top + h, left:left + w].abs().sum()) > 0, "a sample is placed on a blank canvas")


if __name__ == "__main__":
    run(regions, recorder, nf4_probe, int8_probe, artifacts, checks, suites_and_verdict, run_suites, visual_aids)
