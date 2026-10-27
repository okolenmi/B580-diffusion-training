"""The hardware runs of ``TESTING.md`` as scenarios plus checks. Scenario names are the ones ``perf.hwcheck`` already runs, so a suite also judges a
bundle that came back from ``hwcheck`` without running anything. The thresholds carry the measurement or the TESTING item they come from."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..perf.hwcheck import default_scenarios
from ..perf.matrix import Scenario
from .checks import (AdaptersIdentical, Check, GraphDecided, LinesAbsent, LinesPresent, LossCurveClose, LossFinite, NumbersShown, PeakAtLeast, PeakBelow, PeakLower,
                     RunFinished, StepTimeWithin)

NATIVE, INT8 = "native eager", "int8 eager"
NATIVE_G, INT8_G = "native graph", "int8 graph"


@dataclass
class Suite:
    name: str
    title: str
    item: str  # the TESTING.md item this restates
    scenarios: list[Scenario]
    checks: list[Check] = field(default_factory=list)

    def scenario_names(self) -> list[str]:
        return [s.name for s in self.scenarios]


def _named(all_scenarios: list[Scenario], *names: str) -> list[Scenario]:
    by_name = {s.name: s for s in all_scenarios}
    return [by_name[n] for n in names]


def _extra(name: str, window: tuple[str, ...], *args: str, env=()) -> Scenario:
    return Scenario(name, (*window, *args), tuple(env))


class SuiteBook:
    """Builds the suites for one warm-up length and ceilings file (the perf window arguments every scenario carries)."""

    def __init__(self, warmup: int = 20, ceilings: str = "ceilings.json"):
        self.base = default_scenarios(warmup=warmup, ceilings=ceilings)
        self.window = ("--set", f"perf.warmup_steps={warmup}", "--set", f"perf.ceilings={ceilings}")

    def speed(self) -> Suite:
        names = [NATIVE, NATIVE_G, INT8, INT8_G, "int8 dx-int8 eager", "int8 dx-int8 graph", "native graph attention-math_bf16",
                 "native graph attention-chunked_bf16", "int8 graph attention-chunked_bf16",
                 "native eager lora-activation", "native graph attention-math_bf16 lora-activation", "int8 graph attention-math_bf16 lora-activation"]
        scenarios = _named(self.base, *names)
        skip = 10
        checks: list[Check] = [RunFinished(f"{n} finished", (n,)) for n in names]
        checks += [LossFinite(f"{n} loss finite", (n,)) for n in names]
        checks += [
            PeakLower("int8 holds less memory than native", (INT8, NATIVE), 1500, reason="TESTING 23: 2059 MB saved by 652 int8 layers"),
            StepTimeWithin("int8 graph within 15% of native graph", (INT8_G, NATIVE_G), 1.15, skip, reason="TESTING 25 pass criterion"),
            StepTimeWithin("native graph no slower than eager", (NATIVE_G, NATIVE), 1.03, skip, reason="TESTING 26: replay is dropped when it does not pay, the cost is the timed replays"),
            StepTimeWithin("int8 graph no slower than eager", (INT8_G, INT8), 1.03, skip, reason="TESTING 26"),
            GraphDecided("int8 graph decision", (INT8_G,), expect="kept", reason="measured: int8 replay kept with a small gain"),
            GraphDecided("native graph decision", (NATIVE_G,), expect="any", reason="measured: dropped with the math attention; the bf16 attention modes may change it"),
            StepTimeWithin("dense gradient path no slower than int8", (INT8_G, "int8 dx-int8 graph"), 1.02, skip, reason="measured: the int8 input gradient is 4% slower in a graph, so the default stays dense"),
            StepTimeWithin("math_bf16 attention beats math in a graph", ("native graph attention-math_bf16", NATIVE_G), 1.0, skip, reason="TESTING 28"),
            StepTimeWithin("chunked_bf16 attention within 25% of math in a graph", ("native graph attention-chunked_bf16", NATIVE_G), 1.25, skip, reason="measured: 15% slower; it exists to save memory at large sizes, not time"),
            StepTimeWithin("lora in the activation dtype not slower than float32 (eager)", ("native eager lora-activation", NATIVE), 1.04, skip, reason="TESTING 31; eager steps vary by about 3% between identical runs"),
            StepTimeWithin("lora in the activation dtype not slower than float32 (math_bf16 graph)", ("native graph attention-math_bf16 lora-activation", "native graph attention-math_bf16"), 1.04, skip, reason="TESTING 31; graph steps vary by about 5% between identical runs"),
            LossCurveClose("lora activation loss close to float32 (eager)", ("native eager lora-activation", NATIVE), 0.05, 10, reason="TESTING 31: the low-rank intermediate and factor gradients are rounded to bf16"),
            LossCurveClose("int8 loss close to native", (INT8, NATIVE), 0.15, 10, reason="the int8 network is not the native one; a wide band catches divergence, not noise"),
            LossCurveClose("chunked_bf16 loss close to math", ("native graph attention-chunked_bf16", NATIVE_G), 0.10, 10, reason="TESTING 28: scores rounded to bf16"),
        ]
        return Suite("speed", "Speed of native, int8, graph and attention modes", "23, 25, 26, 28", scenarios, checks)

    def memory(self) -> Suite:
        w = self.window
        measured = ("--set", "trainer.tune_checkpointing=true", "--param", "trainer.checkpoint_by=measured")
        int8 = ("--param", "adapter.base_precision=int8")
        scenarios = [_extra("limit 10240 measured native", w, "--set", "memory.vram_max_mb=10240", *measured),
                     _extra("limit 10240 measured int8", w, "--set", "memory.vram_max_mb=10240", *measured, *int8),
                     _extra("limit 9000 int8", w, "--set", "memory.vram_max_mb=9000", *int8)]
        usable = 10240 - 512
        checks: list[Check] = [RunFinished(f"{s.name} finished", (s.name,)) for s in scenarios]
        checks += [LinesPresent(f"{s.name} printed its budget", (s.name,), r"^memory budget:", reason="TESTING 24") for s in scenarios]
        checks += [
            PeakBelow("native stays under the 10240 limit", ("limit 10240 measured native",), 10240, "device_mb", "TESTING 24"),
            PeakBelow("int8 stays under the 10240 limit", ("limit 10240 measured int8",), 10240, "device_mb", "TESTING 24"),
            PeakAtLeast("native uses most of the room", ("limit 10240 measured native",), 0.85 * usable, "device_mb", "TESTING 24: close to the limit minus the margin, not 8.1 GB"),
            PeakAtLeast("int8 uses most of the room", ("limit 10240 measured int8",), 0.85 * usable, "device_mb", "TESTING 24"),
            PeakBelow("9000 limit holds", ("limit 9000 int8",), 9000, "device_mb", "TESTING 24"),
        ]
        return Suite("memory", "Memory budget with one margin", "24", scenarios, checks)

    def nf4(self) -> Suite:
        nf4 = ("--param", "adapter.base_precision=nf4")
        scenarios = [*_named(self.base, NATIVE, INT8), _extra("nf4 eager", self.window, *nf4), _extra("nf4 graph", self.window, "--set", "trainer.graph=true", *nf4),
                     _extra("nf4 double-quant eager", self.window, *nf4, "--param", "adapter.nf4_double_quant=true")]
        checks: list[Check] = [RunFinished(f"{s.name} finished", (s.name,)) for s in scenarios]
        checks += [LinesPresent(f"{n} printed its conversion", (n,), r"^nf4 base weights \(", reason="TESTING 29") for n in ("nf4 eager", "nf4 graph", "nf4 double-quant eager")]
        checks += [LossFinite(f"{s.name} loss finite", (s.name,)) for s in scenarios]
        checks += [
            PeakLower("nf4 holds less memory than int8", ("nf4 eager", INT8), 500, reason="1.3 GB against 2.2 GB of frozen weights, minus the rebuilt layer"),
            PeakLower("double quantization holds less than plain nf4", ("nf4 double-quant eager", "nf4 eager"), 50, reason="about 100 MB on the UNet"),
            LossCurveClose("nf4 loss close to int8", ("nf4 eager", INT8), 0.15, 10, reason="TESTING 29: within the noise of int8's"),
            StepTimeWithin("nf4 not more than 1.6x int8", ("nf4 eager", INT8), 1.6, 10, reason="the weight is rebuilt per call; a wide bound, memory is the point"),
        ]
        return Suite("nf4", "NF4 frozen weights", "29", scenarios, checks)

    def determinism(self) -> Suite:
        det = ("--set", "runtime.deterministic=true", "--output", "{folder}/out")
        scenarios = [_extra("deterministic a", self.window, *det), _extra("deterministic b", self.window, *det)]
        checks: list[Check] = [RunFinished(f"{s.name} finished", (s.name,)) for s in scenarios]
        checks += [LossCurveClose("same losses", ("deterministic a", "deterministic b"), 0.0, 1000, reason="TESTING 22: bit-identical runs"),
                   AdaptersIdentical("same adapter tensors", ("deterministic a", "deterministic b"), reason="TESTING 22")]
        return Suite("determinism", "Deterministic mode", "22", scenarios, checks)

    def allocator(self) -> Suite:
        names = ["native expandable-segments", "native no-expandable-segments"]
        scenarios = _named(self.base, *names)
        checks: list[Check] = [RunFinished(f"{n} finished", (n,)) for n in names]
        checks += [NumbersShown("allocator setting changes the peak?", tuple(names), "device_mb",
                                "Equal peaks mean the setting has no effect on this workload (TESTING 24).", reason="TESTING 24, expandable segments")]
        return Suite("allocator", "Allocator A/B", "24", scenarios, checks)

    def features(self) -> Suite:
        """Short runs of the options TESTING.md lists one by one: each must start, finish, and say what its item expects."""
        def run(name, *args):
            return Scenario(name, args)
        int8 = ("--param", "adapter.base_precision=int8")
        scenarios = [
            run("probe and alignment", "--set", "trainer.probe_every=10", "--set", "trainer.probe_alignment=true", "--set", "trainer.probe_items=4"),
            run("blend max 0.5", "--set", "blend.strength_max=0.5"),
            run("blend fixed 0.5", "--set", "blend.strength_min=0.5", "--set", "blend.strength_max=0.5", "--persist-conditioning"),
            run("cap 9216 no prewarm", "--set", "memory.vram_max_mb=9216", "--set", "conditioning.prewarm=false"),
            run("three settings", "--set", "memory.vram_max_mb=10240", "--set", "trainer.graph=true", "--set", "blend.strength_max=0.5"),
            run("setting typo", "--set", "trainr.graph=true"),
            run("pad zero", "--pad-fill", "zero"),
            run("pad mirror", "--pad-fill", "mirror"),
            run("int8 torch backend", *int8, "--param", "adapter.int8_backend=torch"),
            run("int8 dora", *int8, "--param", "adapter.kind=dora"),
        ]
        scenarios += _named(self.base, INT8)
        done = [s.name for s in scenarios if s.name not in ("setting typo", INT8)]
        checks: list[Check] = [RunFinished(f"{n} finished", (n,)) for n in done]
        checks += [LossFinite(f"{n} loss finite", (n,)) for n in done]
        checks += [
            LinesPresent("probe lines every 10 steps", ("probe and alignment",), r"^  probe: adapter/base loss", 2, "TESTING 7"),
            LinesPresent("gradient alignment after each probe", ("probe and alignment",), r"^  gradient alignment: ", 2, "TESTING 6e"),
            LinesPresent("graph settings take effect next to the others", ("three settings",), r"graphs \d+ \(", 1, "TESTING 20"),
            LinesPresent("a typo in --set names the node", ("setting typo",), r"trainr", 1, "TESTING 20: the run stops at once and lists the node names"),
            LinesPresent("the typo message lists the real names", ("setting typo",), r"trainer", 1, "TESTING 20"),
            LinesAbsent("the typo stops before training", ("setting typo",), r"^step 1/", "TESTING 20"),
            PeakBelow("cap 9216 holds", ("cap 9216 no prewarm",), 9216, "device_mb", "TESTING 11"),
            StepTimeWithin("mirror padding costs about the same as zero", ("pad mirror", "pad zero"), 1.10, 10, reason="measured: 4-6% (the reflection is built per sample)"),
            LinesPresent("int8 summary line (kernels)", (INT8,), r"int8 base weights \(xpu_kernels\): \d+ layers", 1, "TESTING 23"),
            LinesPresent("int8 summary line (torch)", ("int8 torch backend",), r"int8 base weights \(torch\): \d+ layers", 1, "TESTING 23"),
            LinesPresent("int8 torch backend warns", ("int8 torch backend",), r"plain-torch", 1, "TESTING 23"),
        ]
        return Suite("features", "Options listed one by one in TESTING.md", "6c, 6e, 7, 11, 17, 20, 23", scenarios, checks)

    def all(self) -> dict[str, Suite]:
        return {s.name: s for s in (self.speed(), self.memory(), self.nf4(), self.determinism(), self.allocator(), self.features())}
