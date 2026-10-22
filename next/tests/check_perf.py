"""Performance lab: kernel categories, cost and rating arithmetic, findings rules on synthetic reports, and a profiling window
over the tiny trainer (CPU: kernel times are host operator times, so only the structure and the sums are checked).

    python -m next.tests.check_perf
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from torch import nn

from ..perf.categories import Categorizer
from ..perf.ceilings import Ceilings
from ..perf.cost import CostModel
from ..perf.findings import findings_for
from ..perf.modules import region_of, role_of
from ..perf.rating import Rating
from ..perf.session import PerfSession
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
        check(report["budget"]["steps"] == 2 and report["meta"]["steps_in_window"] == 2 and session.records[0].step == 2, "only the window's steps are profiled, after the warm-up")
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


def short_run():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "report"
        out = rig(Path(tmp), steps=3, perf=None)
        trainer, denoiser = out[0], out[2]
        said = []
        trainer._perf = PerfSession(denoiser.module(), "cpu", folder, warmup=10, steps=2, quick_ceilings=True, say=said.append)
        trainer.run()
        check(not (folder / "report.md").exists() and any("before the window" in s for s in said), "a run shorter than the warm-up says so and writes nothing")
        out = rig(Path(tmp) / "b", steps=4, perf=None)
        trainer, denoiser = out[0], out[2]
        trainer._perf = PerfSession(denoiser.module(), "cpu", folder, warmup=2, steps=5, quick_ceilings=True, say=lambda text: None)
        trainer.run()
        check((folder / "report.md").exists(), "a run that ends inside the window still reports the steps it got")
        raises(ValueError, lambda: PerfSession(denoiser.module(), "cpu", folder, steps=0), "an empty window is refused", contains="at least 1")


if __name__ == "__main__":
    run(categories, cost_and_rating, rules, window, short_run)
