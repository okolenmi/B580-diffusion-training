"""Performance lab: kernel categories, cost and rating arithmetic, findings rules on synthetic reports, and a profiling window
over the tiny trainer (CPU: kernel times are host operator times, so only the structure and the sums are checked).

    python -m next.tests.check_perf
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from pathlib import Path

import torch
from torch import nn
from torch.autograd import DeviceType

from types import SimpleNamespace

from ..graph import ExecutionContext, SettingsRegistry
from ..nodes.train.trainer import DiffusionTrainerNode
from ..perf import bench as bench_module
from ..perf.budget import StepRecord
from ..perf.bench import BenchRunner, context_for
from ..perf.categories import Categorizer
from ..perf.ceilings import Ceilings
from ..perf.cost import CostModel
from ..perf.comparison import Comparison
from ..perf.findings import findings_for
from ..perf.hwcheck import HwCheck, StepResult, TrainArguments, default_scenarios
from ..perf.matrix import Matrix, MatrixSpec, Outcome, Scenario
from ..perf.modules import region_of, role_of
from ..perf.power import PowerMeter
from ..perf.rating import Rating
from ..perf.session import PerfSession
from ..perf.trace import KernelTrace
from .check_engine import rig
from .support import check, raises, run

CEILINGS = Ceilings(device="xpu", gemm_tflops=50.0, int8_tops=100.0, copy_gbps=400.0, dispatch_us=5.0, launch_us=8.0)


def categories():
    c = Categorizer()
    check(c.category("gemm_kernel<bf16>") == "gemm" and c.category("aten::convolution") == "convolution", "gemm and convolution names")
    check(c.category("xpu_int8_convrot_quantize") == "int8", "int8 rules win over the convolution rule")
    check(c.category("softmax_forward") == "attention" and c.category("flash_fwd") == "attention", "attention kernels")
    check(c.category("aten::group_norm") == "norm" and c.category("aten::_to_copy") == "copy", "norm and copy")
    check(c.category("totally_new_thing") == "other" and c.unknown == {"totally_new_thing": 1}, "an unknown name is 'other' and remembered")


def cost_and_rating():
    model = CostModel()
    linear = nn.Linear(64, 32)
    cost = model.cost("linear", linear, [(2, 10, 64)], (2, 10, 32), 2)
    check(cost.flops == 2 * 20 * 32 * 64 and cost.bytes == (20 * 64 + 20 * 32 + 32 * 64) * 2 and not cost.low_precision, "linear flops and bytes")
    conv = nn.Conv2d(8, 16, 3, padding=1)
    cost = model.cost("conv", conv, [(1, 8, 4, 4)], (1, 16, 4, 4), 4)
    check(cost.flops == 2 * 16 * 16 * 8 * 9, "conv flops")
    check(model.cost("block", linear, [(1, 2)], (1, 2), 2) is None and model.cost("linear", linear, [], (1,), 2) is None, "unknown kinds and missing shapes are unrated")
    rating = Rating(CEILINGS)
    ideal, bound = rating.ideal_ms(type(cost)(flops=50e12 * 0.001, bytes=1.0))
    check(abs(ideal - 1.0) < 1e-9 and bound == "compute", "a matmul sized to one millisecond at the ceiling")
    ideal, bound = rating.ideal_ms(type(cost)(flops=1.0, bytes=400e9 * 0.002))
    check(abs(ideal - 2.0) < 1e-9 and bound == "memory", "memory-bound work is rated against the copy ceiling")
    fast, _ = rating.ideal_ms(type(cost)(flops=1e12, bytes=1.0, low_precision=True))
    slow, _ = rating.ideal_ms(type(cost)(flops=1e12, bytes=1.0))
    check(fast < slow, "int8 matmuls use the int8 ceiling")
    rated = rating.rate(4.0, 1.0, "compute")
    check(rated.recoverable_ms == 3.0 and rated.efficiency == 0.25, "recoverable time and efficiency")
    check(rating.rate(0.5, 1.0, "compute").recoverable_ms == 0.0, "faster than ideal recovers nothing")
    check(region_of("input_blocks.1.0.in_layers.2") == "down" and role_of("input_blocks.1.1.transformer_blocks.0.attn1.to_q") == "attn1.to_q", "region and role from a path")


def rules():
    base = {"budget": {"device_is_host": False, "utilisation": 0.4, "device_ms": 400, "wall_ms": 1000, "launches": 9000, "ops": 12000, "host_us_per_launch": 90,
                       "replayed_ms": None, "outliers": []},
            "categories": [{"category": "gemm", "calls": 100, "device_ms": 300.0}, {"category": "attention", "calls": 10, "device_ms": 60.0},
                           {"category": "elementwise", "calls": 500, "device_ms": 40.0}],
            "kernels": [{"name": "softmax_forward", "category": "attention", "device_ms": 30.0}], "modules": [], "groups": [], "reconciliation": {}}
    titles = {f.title for f in findings_for(base)}
    check("The step is launch bound" in titles and "Attention runs as math (matmul, softmax, matmul)" in titles, "launch-bound and math attention are found")
    healthy = {**base, "budget": {**base["budget"], "utilisation": 0.9}, "kernels": [{"name": "flash_fwd", "category": "attention", "device_ms": 30.0}]}
    check(not findings_for(healthy), "a healthy step produces no findings")
    host = {**base, "budget": {**base["budget"], "device_is_host": True}}
    check("The step is launch bound" not in {f.title for f in findings_for(host)}, "utilisation is not judged from host operator times")
    slow = {**healthy, "groups": [{"kind": "linear", "role": "attn1.to_q", "region": "down", "count": 4, "total_ms": 20.0, "ideal_ms": 5.0, "recoverable_ms": 15.0,
                                   "efficiency": 0.25, "bound": "compute"}]}
    found = findings_for(slow)
    check(len(found) == 1 and found[0].severity_ms == 15.0 and "attn1.to_q" in found[0].title, "the slowest groups are named")
    mods = {**healthy, "modules": [{"fwd_ms": 10.0, "recompute_ms": 20.0, "bwd_ms": 30.0}]}
    check("Recomputing activations is a large part of the step" in {f.title for f in findings_for(mods)}, "recompute share")
    outliers = {**healthy, "budget": {**healthy["budget"], "outliers": [{"step": 9, "shape": "(1, 64, 64)", "wall_ms": 5000.0}]}}
    check("Some steps in the window were much slower than the rest" in {f.title for f in findings_for(outliers)}, "slow steps are reported")


def raw_events():
    """The rows come from the profiler's raw events: counts, durations and operator self time agree with the averaged table."""
    x, lin = torch.randn(16, 16), nn.Linear(16, 16)
    trace = KernelTrace("cpu")
    with trace:
        for _ in range(20):
            lin(x).sum().backward()
    averaged = trace._collect_averaged()
    fast = trace.result
    names = {r.name: r for r in fast.rows}
    check(fast.device_is_host and fast.ops > 0 and {"aten::addmm", "aten::sum"} <= set(names), "a CPU run reports host operators")
    check(abs(fast.host_op_us - averaged.host_op_us) <= 0.01 * averaged.host_op_us + 1.0, "operator self time agrees with the averaged table")
    check(set(trace.result.seconds) >= {"profiler stop", "events to rows"}, "the cost of reading the profile is recorded")


class FakeEvent:
    """The accessors of a profiler event that the attribution reads."""

    def __init__(self, name, device="cpu", activity="cpu_op", start=0, end=0, thread=1, corr=0, linked=0, shapes=(), dtypes=()):
        self._v = dict(name=name, start_ns=start, end_ns=end, duration_ns=end - start, start_thread_id=thread, correlation_id=corr,
                       linked_correlation_id=linked, shapes=[list(x) for x in shapes], dtypes=list(dtypes), activity_type=activity)
        self._device = DeviceType.CPU if device == "cpu" else DeviceType.XPU

    def device_type(self):
        return self._device

    def is_hidden_event(self):
        return False

    def __getattr__(self, field):
        if field.startswith("_"):
            raise AttributeError(field)
        return lambda: self._v[field]


def provenance():
    """A kernel's linked correlation id names the operator that launched it; its chain and module range come from where that operator ran."""
    from ..perf.categories import Categorizer
    from ..perf.events import OpTree, read_events
    from ..perf.provenance import ProvenanceBuilder, matmul_flops

    events = [
        FakeEvent("perf|input_blocks.1.0.attn|forward", activity="user_annotation", start=0, end=5050),
        FakeEvent("aten::linear", start=100, end=5000),
        FakeEvent("aten::addmm", start=200, end=4000, corr=7, shapes=[[64], [128, 32], [32, 64]], dtypes=["c10::BFloat16"] * 3),
        FakeEvent("zeKernelSetArgumentValue", activity="xpu_driver", start=300, end=320, corr=8, linked=7),  # driver ids collide with operator ids
        FakeEvent("gemm_kernel", device="xpu", activity="kernel", start=9000, end=59_000, linked=7),
        FakeEvent("aten::to", start=5100, end=6000),
        FakeEvent("aten::copy_", start=5200, end=5900, corr=8, shapes=[[8, 8], [8, 8]], dtypes=["c10::BFloat16", "float"]),
        FakeEvent("at::native::xpu::CopyScalarFunc", device="xpu", activity="kernel", start=60_000, end=90_000, linked=8),
        FakeEvent("gemm_kernel", device="xpu", activity="kernel", start=91_000, end=141_000, linked=0),  # a replayed kernel: no launch recorded
    ]
    table = read_events(events, keep_raw=True)
    tree = OpTree(table.ops)
    result = ProvenanceBuilder(table, tree, Categorizer()).build()
    rows = {r["kernel"]: r for r in result.rows if r["op"]}
    gemm, cast = rows["gemm_kernel"], rows["at::native::xpu::CopyScalarFunc"]
    check(gemm["op"] == "aten::linear > aten::addmm" and gemm["shapes"] == "64, 128x32, 32x64" and gemm["flops"] == 2 * 128 * 32 * 64, "a matrix kernel gets its operator chain, shapes and FLOPs")
    check(cast["op"] == "aten::to > aten::copy_" and cast["dtypes"] == "bf16,f32" and cast["module"] == "", "a cast gets its operator chain and dtypes; outside a module range it has no module")
    check(gemm["module"] == "input_blocks.attn" and result.modules["input_blocks.1.0.attn"]["forward"] == [50.0, 50.0], "kernel time is added to the module range it was launched in")
    check(abs(result.coverage["operator"] - 80 / 130) < 1e-9 and any(r["op"] == "" for r in result.rows), "kernels without a launch stay unattributed and the coverage says how many")
    check(matmul_flops("aten::bmm", [[4, 8, 16], [4, 16, 32]]) == 2 * 4 * 8 * 16 * 32 and matmul_flops("aten::relu", [[2, 2]]) is None, "FLOPs of batched matrix products, none for other operators")
    tree_ops = {op.name: own for op, own in zip(tree.ops, tree.own_us)}
    check(abs(tree_ops["aten::linear"] - 1.1) < 1e-9, "operator self time excludes the operators called inside")
    report = {"categories": [{"category": "copy", "device_ms": 30.0}, {"category": "gemm", "device_ms": 70.0}],
              "provenance": [{"category": "copy", "op": "aten::to > aten::copy_", "module": "attn", "device_ms": 30.0, "calls": 400, "dtypes": "bf16,f32", "shapes": "8x8, 8x8", "tflops": None, "efficiency": None},
                             {"category": "gemm", "op": "aten::mm", "module": "ff", "device_ms": 20.0, "calls": 10, "dtypes": "bf16", "shapes": "9x9, 9x9", "tflops": 20.0, "efficiency": 0.2}],
              "meta": {"provenance_coverage": {"operator": 0.1}, "profiler_event_kinds": {}}}
    from ..perf.findings import AttributionCoverage, CopySources, GemmShapes
    titles = {f.title for rule in (CopySources(), GemmShapes(), AttributionCoverage()) for f in rule.check(report)}
    check(any("Copy kernels come from aten::to > aten::copy_" in t for t in titles) and any("runs at 20% of the ceiling" in t for t in titles) and "Kernels could not be tied to operators" in titles,
          "the provenance rules name the cast source, the slow matrix shape and a failed attribution")


def capture_is_skipped():
    """Module events are not recorded while a graph is captured: the capture runs the module without running its kernels."""
    from ..perf.clock import DeviceClock
    from ..perf.modules import ModuleTimer

    clock = DeviceClock("cpu")
    state = {"capturing": False}
    clock.capturing = lambda: state["capturing"]
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    timer = ModuleTimer(model, clock).attach()
    x = torch.randn(2, 4, requires_grad=True)
    model(x).sum().backward()
    state["capturing"] = True
    model(x).sum().backward()
    state["capturing"] = False
    timer.detach()
    records = {r.path: r for r in timer.records()}
    check(records and all(r.calls == 1 for r in records.values()), "a pass made while capturing is not counted")
    check(DeviceClock("cpu").capturing() is False, "a CPU clock is never capturing")


def window():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "report"
        session = None

        def build(steps):
            nonlocal session
            out = rig(Path(tmp), steps=steps, perf=None)
            trainer, denoiser = out[0], out[2]
            session = PerfSession(denoiser.module(), "cpu", folder, warmup=2, steps=2, quick_ceilings=True, say=lambda text: None)
            trainer._perf = session
            return trainer

        build(6).run()
        check(session.report is not None and session.done, "the window closed and the report was built")
        check((folder / "report.md").exists() and (folder / "report.json").exists() and (folder / "ceilings.json").exists(), "report files and ceilings are written")
        report = json.loads((folder / "report.json").read_text())
        check(report["budget"]["steps"] == 2 and report["meta"]["steps_in_window"] == 2 and [r.step for r in session.clean] == [2, 3] and [r.step for r in session.records] == [4, 5],
              "after the warm-up, steps are first timed alone and then profiled")
        check(report["budget"]["wall_clean"] and report["budget"]["timed_steps"] == 2 and "profiler stop" in report["meta"]["report_seconds"], "the step time comes from the unprofiled steps and the report says what it cost")
        check(all(r.wall_ms > 0 and r.shape and r.shape[0] for r in session.records), "each step has a wall time and a shape")
        kinds = {m["kind"] for m in report["modules"]}
        check({"linear", "conv"} <= kinds, f"modules of the tiny UNet are timed ({sorted(kinds)})")
        check(all(m["fwd_ms"] >= 0 and m["bwd_ms"] >= 0 for m in report["modules"]) and any(m["bwd_ms"] > 0 for m in report["modules"]), "forward and backward are recorded")
        rated = [m for m in report["modules"] if m["ideal_ms"] is not None]
        check(rated and all(m["recoverable_ms"] >= 0 and 0 <= m["efficiency"] for m in rated), "rated modules carry ideal, recoverable time and efficiency")
        check(report["groups"] and abs(sum(g["total_ms"] for g in report["groups"]) - sum(m["self_ms"] for m in rated)) < 1e-6, "groups add up to their modules")
        rec = report["reconciliation"]
        check(rec["module_ms"] is not None and rec["unrated_ms"] <= rec["module_ms"] + 1e-9, "reconciliation reports module and unrated time")
        check(report["meta"]["device_time_is_host_time"] and report["budget"]["utilisation"] is None, "a CPU run says its times are host times")
        check(abs(sum(c["share"] for c in report["categories"]) - 1.0) < 1e-6, "category shares add up")
        text = (folder / "report.md").read_text()
        check("## Step budget" in text and "## Slowest piece groups" in text and "## Slowest kernels" in text, "the markdown has the sections")
        check(not any(m._forward_hooks or m._backward_hooks for m in session.model.modules()), "hooks are removed after the window")


def steady_steps():
    """A step is steady unless it met a new shape, captured a graph, or ran eagerly in a graph run."""
    session = PerfSession(nn.Linear(2, 2), "cpu", "unused", warmup=0, steps=1, quick_ceilings=True, say=lambda text: None)
    sequence = [  # (shape, replays, graphs, steady)
        ((2, 8, 8), 0, 0, False),   # a new shape, eager
        ((2, 8, 8), 0, 0, False),   # eager step in a graph run
        ((2, 8, 8), 1, 1, False),   # the capture step
        ((2, 8, 8), 2, 1, True),    # a replay
        ((2, 6, 6), 2, 1, False),   # another new shape
        ((2, 8, 8), 3, 1, True),    # back to the captured one
    ]
    got = []
    for n, (shape, replays, graphs, _) in enumerate(sequence):
        session._current = StepRecord(n, 1.0)
        session.observe({"batch": shape[0], "latent_h": shape[1], "latent_w": shape[2], "graph_replays": replays, "graphs": graphs})
        got.append(session._current.steady)
    check(got == [row[3] for row in sequence], f"steady steps are told from new shapes, captures and eager steps ({got})")
    eager = PerfSession(nn.Linear(2, 2), "cpu", "unused", warmup=0, steps=1, quick_ceilings=True, say=lambda text: None)
    flags = []
    for shape in ((2, 8, 8), (2, 8, 8), (2, 6, 6)):
        eager._current = StepRecord(0, 1.0)
        eager.observe({"batch": shape[0], "latent_h": shape[1], "latent_w": shape[2]})
        flags.append(eager._current.steady)
    check(flags == [False, True, False], "without graphs only a new shape is unsteady")
    session._current = StepRecord(9, 1.0, eager_profile=True)
    session.observe({"batch": 2, "latent_h": 8, "latent_w": 8, "graph_replays": 3, "graphs": 1})
    check(session._current.steady, "a profiled step forced eager in a graph run is steady when its shape is known and nothing was captured")


def window_provenance():
    """A profiled window with provenance on (a CPU run has no kernels, so the tables are empty) still writes a complete report."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "report"
        out = rig(Path(tmp), steps=6, perf=None)
        trainer, denoiser = out[0], out[2]
        session = PerfSession(denoiser.module(), "cpu", folder, warmup=2, steps=2, quick_ceilings=True, provenance=True, say=lambda text: None)
        trainer._perf = session
        trainer.run()
        report = json.loads((folder / "report.json").read_text())
        check(report["provenance"] == [] and report["meta"]["module_times"] == "span" and report["regions"], "without kernels the module spans and regions are reported")
        check(not any(m._forward_hooks or m._backward_hooks for m in session.model.modules()), "hooks are removed after the window")


def profile_failure():
    """When the profiler's own post-processing fails, the step times already measured are still reported, with the reason."""
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "report"
        out = rig(Path(tmp), steps=6, perf=None)
        trainer, denoiser = out[0], out[2]
        original = KernelTrace._collect

        def broken(self):
            raise MemoryError("std::bad_array_new_length")
        KernelTrace._collect = broken
        try:
            trainer._perf = PerfSession(denoiser.module(), "cpu", folder, warmup=2, steps=2, quick_ceilings=True, say=lambda text: None)
            trainer.run()
        finally:
            KernelTrace._collect = original
        report = json.loads((folder / "report.json").read_text())
        check("bad_array_new_length" in report["meta"]["profile_error"] and report["budget"]["wall_clean"], "the report keeps the step time and names the profiler failure")


def short_run():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "report"
        out = rig(Path(tmp), steps=3, perf=None)
        trainer, denoiser = out[0], out[2]
        said = []
        trainer._perf = PerfSession(denoiser.module(), "cpu", folder, warmup=10, steps=2, quick_ceilings=True, say=said.append)
        trainer.run()
        check(not (folder / "report.md").exists() and any("before the profiled window" in s for s in said), "a run shorter than the warm-up says so and writes nothing")
        out = rig(Path(tmp) / "b", steps=4, perf=None)
        trainer, denoiser = out[0], out[2]
        trainer._perf = PerfSession(denoiser.module(), "cpu", folder, warmup=1, steps=2, quick_ceilings=True, say=lambda text: None)
        trainer.run()
        check((folder / "report.md").exists(), "a run that ends inside the window still reports the steps it got")
        raises(ValueError, lambda: PerfSession(denoiser.module(), "cpu", folder, steps=0), "an empty window is refused", contains="at least 1")


def synthetic(wall: float, gemm: float, copy: float, linear_ms: float) -> dict:
    return {"budget": {"wall_ms": wall, "eager_ms": wall, "replayed_ms": None, "device_ms": gemm + copy, "launches": 100.0, "utilisation": 0.8, "peak_mb": 5000.0,
                       "device_is_host": False, "outliers": []},
            "categories": [{"category": "gemm", "calls": 10.0, "device_ms": gemm, "share": gemm / (gemm + copy)},
                           {"category": "copy", "calls": 50.0, "device_ms": copy, "share": copy / (gemm + copy)}],
            "kernels": [], "modules": [], "reconciliation": {},
            "groups": [{"kind": "linear", "role": "attn1.to_q", "region": "down", "count": 4, "total_ms": linear_ms, "ideal_ms": 1.0, "recoverable_ms": linear_ms - 1.0,
                        "efficiency": 1.0 / linear_ms, "bound": "compute"}],
            "findings": []}


def comparison():
    diff = Comparison(synthetic(1000, 700, 50, 10.0), synthetic(1300, 700, 350, 25.0), "native", "int8")
    cats = diff.categories()
    check(cats[0]["category"] == "copy" and abs(cats[0]["delta_ms"] - 300) < 1e-9, "the category that grew most comes first")
    check(diff.groups()[0]["delta_ms"] == 15.0 and diff.budget()[0]["delta"] == 300, "group and budget deltas")
    text = diff.markdown()
    check("int8 against native" in text and "+300" in text and "attn1.to_q" in text, "the markdown names both runs and the pieces that changed")
    one_sided = Comparison(synthetic(1, 1, 1, 2.0), {**synthetic(1, 1, 1, 2.0), "groups": []}, "a", "b").groups()
    check(one_sided[0]["only_in"] == "a", "a group present in one run only is marked")


SPEC = {"base": {"command": ["train", "--x"]}, "axes": {"precision": {"native": [], "int8": ["--param", "p=int8"]}, "graph": {"eager": [], "graph": ["--set", "g=true"]}},
        "scenario": [{"name": "extra one", "args": ["--y"], "env": {"A": "1"}}]}


class FakeRunner:
    def __init__(self, fail: str | None = None, silent: str | None = None):
        self.fail, self.silent, self.ran = fail, silent, []

    def run(self, scenario, folder):
        self.ran.append(scenario.name)
        folder.mkdir(parents=True, exist_ok=True)
        if scenario.name == self.fail:
            return Outcome(scenario, False, 1.0, folder, error="exit code 1")
        if scenario.name != self.silent:
            (folder / "report.json").write_text(json.dumps(synthetic(1000 + 100 * len(self.ran), 700, 50 + 10 * len(self.ran), 10.0)))
        return Outcome(scenario, True, 1.0, folder)


def power():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        check(PowerMeter(root).counter is None and PowerMeter(root).stop() is None, "no card, no reading")
        other = root / "card0" / "device"
        (other / "hwmon" / "hwmon1").mkdir(parents=True)
        (other / "vendor").write_text("0x10de\n")
        (other / "hwmon" / "hwmon1" / "energy1_input").write_text("0")
        check(PowerMeter(root).counter is None, "another vendor's card is not measured")
        counter = root / "card1" / "device" / "hwmon" / "hwmon3" / "energy1_input"
        counter.parent.mkdir(parents=True)
        (root / "card1" / "device" / "vendor").write_text("0x8086\n")
        counter.write_text("1000000")
        now = [10.0]
        meter = PowerMeter(root, clock=lambda: now[0])
        check(meter.counter == counter, "the Intel card's energy counter is found")
        meter.start()
        now[0], _ = 12.0, counter.write_text("161000000")
        check(abs(meter.stop() - 80.0) < 1e-9, "average watts is energy over time")
        check(meter.stop() is None, "a stopped meter reports nothing")
        meter.start()
        counter.write_text("5")
        check(meter.stop() is None, "a counter that went backwards is not trusted")


def matrix():
    spec = MatrixSpec.from_dict(SPEC)
    names = [s.name for s in spec.scenarios]
    check(names == ["precision=native,graph=eager", "precision=native,graph=graph", "precision=int8,graph=eager", "precision=int8,graph=graph", "extra one"], "axes multiply, hand-written scenarios follow")
    check(spec.scenarios[3].args == ("--param", "p=int8", "--set", "g=true") and dict(spec.scenarios[4].env) == {"A": "1"}, "arguments and environment per scenario")
    raises(ValueError, lambda: MatrixSpec.from_dict({"axes": {}}), "a file without a command is refused", contains="base.command")
    raises(ValueError, lambda: MatrixSpec.from_dict({"base": {"command": ["x"]}}), "a file without scenarios is refused", contains="no scenarios")
    raises(ValueError, lambda: MatrixSpec.from_dict({"base": {"command": ["x"]}, "scenario": [{"name": "a b"}, {"name": "a_b"}]}), "names that collide as folders are refused", contains="differ")
    with tempfile.TemporaryDirectory() as tmp:
        runner = FakeRunner(fail="precision=int8,graph=eager", silent="extra one")
        path = Matrix(spec, Path(tmp), runner, say=lambda text: None).run()
        text = path.read_text()
        check(len(runner.ran) == 5 and "FAILED: exit code 1" in text, "a failed scenario is recorded and the rest still run")
        check("wrote no report" in text, "a run that ends without a report says so")
        check("precision=int8,graph=graph against precision=native,graph=eager" in text, "every scenario is compared with the first")
        again = FakeRunner()
        Matrix(spec, Path(tmp), again, resume=True, say=lambda text: None).run()
        check(again.ran == ["precision=int8,graph=eager", "extra one"], "resuming reruns only the scenarios without a report")


def micro():
    with tempfile.TemporaryDirectory() as tmp:
        ctx = context_for("cpu", True, tmp)
        runner = BenchRunner(ctx, ["gemm", "conv", "norm", "elementwise", "launch", "attention", "layers"], say=lambda text: None)
        report = runner.run()
        groups = {r.group for r in report.rows}
        check(not report.errors and {"gemm", "convolution", "norm", "elementwise", "attention", "launch", "linear layer"} <= groups, f"every suite produced rows ({sorted(groups)}, errors {list(report.errors)})")
        timed = [r for r in report.rows if r.ms is not None and r.ideal_ms is not None]
        check(timed and all(r.ms > 0 and r.ideal_ms >= 0 for r in timed), "timed rows carry an ideal time")
        check(any(r.note.startswith("unavailable") for r in report.rows if r.group == "attention") or all(r.ms is not None for r in report.rows if r.group == "attention"), "a missing attention backend is a row, not a crash")
        check(any("no graph API" in r.note for r in report.rows if r.group in ("launch", "attention")), "without a graph API the capture rows say so")
        path = runner.write(report, Path(tmp) / "out")
        check(path.exists() and (path.parent / "bench.json").exists() and "## gemm" in path.read_text(), "bench.md and bench.json are written")
        raises(ValueError, lambda: BenchRunner(ctx, ["nonsense"]), "an unknown suite is refused", contains="nonsense")
        saved = dict(bench_module.SUITES)
        try:
            class Broken(bench_module.Bench):
                def rows(self, ctx):
                    raise RuntimeError("boom")
            bench_module.SUITES["broken"] = Broken
            failed = BenchRunner(ctx, ["broken", "launch"], say=lambda text: None).run()
        finally:
            bench_module.SUITES.clear()
            bench_module.SUITES.update(saved)
        check("broken" in failed.errors and any(r.group == "launch" for r in failed.rows) and "Suites that failed" in failed.markdown(), "a failing suite is recorded and the others still run")


def train_arguments():
    import argparse
    import subprocess
    import sys
    from ..perf.hwcheck import TrainArguments
    parser = argparse.ArgumentParser()
    TrainArguments.add_to(parser)
    args, rest = TrainArguments.parse(parser, ["--models", "M", "--datasets", "D", "--dataset", "set", "--file", "f.safetensors", "--output", "O", "--batch", "2", "--", "--param", "buckets.max_shapes=4"])
    out = TrainArguments.from_args(args, rest)
    check(out == ["--models", "M", "--datasets", "D", "--dataset", "set", "--file", "f.safetensors", "--output", "O", "--batch", "2", "--param", "buckets.max_shapes=4"],
          f"named options and the extras after -- become the trainer's arguments ({out})")
    raises(SystemExit, lambda: TrainArguments.parse(parser, ["--models", "M"]), "the options that say what to train are required")
    command = HwCheck(Path("x"), out, steps=12).train_command()
    check(command[-2:] == ["--steps", "12"] and command[1:4] == ["-m", "next.run", "sdxl-lora"], "the step count is added by hwcheck after the user's arguments")
    from ..accept.acceptance import Acceptance
    check(Acceptance([], Path("x"), out, steps=12).command[1:4] == command[1:4], "the acceptance lab runs the same subcommand")
    done = subprocess.run([*command[:4], "--help"], capture_output=True, text=True)
    check(done.returncode == 0 and "--models" in done.stdout, "that subcommand is one next.run really has")


def node_session():
    """The trainer node reads the perf settings through its declared view, the way a real run does."""
    model = SimpleNamespace(denoiser=SimpleNamespace(module=lambda: nn.Linear(2, 2)), device=torch.device("cpu"), info={"base_precision": "nf4"})

    def node(values):
        context = ExecutionContext(settings=SettingsRegistry().snapshot(values)).scoped("trainer", DiffusionTrainerNode.READS_SETTINGS, "Diffusion Trainer")
        return DiffusionTrainerNode(context)
    check(node({})._perf_session(model, None) is None, "without a report folder the node profiles nothing")
    with tempfile.TemporaryDirectory() as tmp:
        session = node({"perf.dir": tmp, "perf.warmup_steps": 5, "perf.steps": 2})._perf_session(model, None)
        check(session is not None and session.warmup == 5 and session.steps == 2, "the node passes the declared settings to the session")


def option_order():
    """Options work in any place: hwcheck's own after trainer options (with or without --), trainer options before them, no abbreviation swallowing a trainer option."""
    from ..perf.__main__ import build_parser
    parser = build_parser()
    mine = ["--models", "M", "--datasets", "D", "--dataset", "n", "--file", "f", "--output", "O"]
    for order in (["--only", "events", "matrix", "--scenarios", "int8 eager", *mine, "--", "--param", "a=1"],
                  [*mine, "--param", "a=1", "--only", "events", "matrix", "--scenarios", "int8 eager"],
                  [*mine, "--", "--param", "a=1", "--only", "events", "matrix", "--scenarios", "int8 eager"]):
        args, rest = TrainArguments.parse(parser, ["hwcheck", *order])
        check(args.only == ["events", "matrix"] and args.scenarios == ["int8 eager"] and TrainArguments.from_args(args, rest)[-2:] == ["--param", "a=1"],
              f"the same arguments in another order mean the same ({order[:2]}...)")
    args, rest = TrainArguments.parse(parser, ["hwcheck", *mine, "--set", "x=1", "--rank", "16", "--step", "5"])
    check(rest == ["--set", "x=1", "--rank", "16", "--step", "5"] and args.steps == 28, "an unknown option that starts like a known one stays the trainer's")


def hwcheck():
    scenarios = default_scenarios(warmup=20, ceilings="c.json")
    picked = HwCheck(Path("x"), [], scenarios=["int8 eager", "native_graph"], warmup=20).scenarios()
    check([s.name for s in picked] == ["native eager", "native graph", "int8 eager"], "a scenario filter keeps the baseline and the named ones, by name or folder name")
    raises(ValueError, lambda: HwCheck(Path("x"), [], scenarios=["nonsense"]).scenarios(), "an unknown scenario name is refused with the list", "nonsense")
    check(scenarios[0].name == "native eager" and len({s.slug for s in scenarios}) == len(scenarios), "the baseline comes first and every scenario has its own folder")
    check(all("perf.ceilings=c.json" in s.args and "perf.warmup_steps=20" in s.args for s in scenarios), "every scenario shares the ceilings cache and the warm-up")
    check(any(dict(s.env).get("PYTORCH_ALLOC_CONF") == "expandable_segments:False" for s in scenarios), "the allocator A/B is among the scenarios")
    with tempfile.TemporaryDirectory() as tmp:
        ran = []

        def fake(step, log):
            ran.append(step.name)
            log.write_text("$ " + " ".join(step.command) + "\n" + ("x" * 400_000 if step.name == "bench" else ""))
            (log.parent.parent / "trace.json").write_text("{}")
            return StepResult(step.name, step.name != "allocator plain", 0.1, "exit code 1" if step.name == "allocator plain" else "")
        out = Path(tmp) / "bundle"
        index = HwCheck(out, ["--file", "m.safetensors"], run_step=fake, say=lambda text: None, only={"hardware", "allocator", "bench"}).run()
        check(ran == ["hardware", "allocator expandable", "allocator plain", "bench"], "the steps run in order, each as its own process")
        text = index.read_text()
        check("FAILED: exit code 1" in text and "bundle.zip" in text and "verdict/verdict.md" in text, "the index lists failures and names the bundle and the verdict")
        with zipfile.ZipFile(out / "bundle.zip") as archive:
            names = archive.namelist()
            big = archive.read("logs/bench.log")
        check("README.md" in names and "logs/hardware.log" in names and not any(n.endswith("trace.json") for n in names), "the zip holds text files, no traces")
        check(len(big) < 210_000 and b"middle cut" in big, "a huge log is cut in the middle")


if __name__ == "__main__":
    run(categories, cost_and_rating, rules, raw_events, provenance, capture_is_skipped, window, steady_steps, window_provenance, profile_failure, short_run, comparison, power, matrix, micro, train_arguments, option_order, node_session, hwcheck)
