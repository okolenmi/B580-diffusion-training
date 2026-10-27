"""One command for the owner's pending hardware runs: every step in its own process (fresh allocator, nothing held by this one), one folder,
one zip of text files to send back. A step that fails is recorded and the rest still run."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

from .bundle import pack
from .matrix import Matrix, MatrixSpec, Scenario, SubprocessRunner



class TrainArguments:
    """The arguments that say *what* to train, shared by ``hwcheck`` and the acceptance lab, turned into the arguments of ``python -m next.run sdxl-lora``."""

    REQUIRED = (("models", "FOLDER", "the folder that holds your models; checkpoints are looked up in FOLDER/checkpoints (or in FOLDER itself when it has no such subfolder)"),
                ("datasets", "FOLDER", "the folder where your datasets are stored (the one you gave to `python -m next.run ingest --datasets`)"),
                ("dataset", "NAME", "the name of one dataset inside that folder (the NAME you gave to `ingest`)"),
                ("file", "NAME", "the checkpoint's file name, relative to FOLDER/checkpoints, e.g. sd_xl_base_1.0.safetensors"),
                ("output", "FOLDER", "where trained adapters are saved"))
    OPTIONAL = (("batch", "N", "images per step"), ("resolution", "PIXELS", "training resolution, e.g. 512"))

    @classmethod
    def add_to(cls, parser) -> None:
        for name, metavar, text in cls.REQUIRED:
            parser.add_argument(f"--{name}", required=True, metavar=metavar, help=text)
        for name, metavar, text in cls.OPTIONAL:
            parser.add_argument(f"--{name}", metavar=metavar, help=text)

    @staticmethod
    def parse(parser, argv: list[str] | None) -> tuple[object, list[str]]:
        """Options may come in any order and place. The ones the parser knows are its own; every other token is the trainer's
        (any other `python -m next.run sdxl-lora` option, e.g. --rank 16 --param buckets.max_shapes=4), in the order given.
        A lone -- changes nothing, so it is dropped. Callers build their parsers with ``allow_abbrev=False``: an abbreviation
        would swallow a trainer option that happens to start like one of theirs."""
        tokens = [a for a in (sys.argv[1:] if argv is None else argv) if a != "--"]
        return parser.parse_known_args(tokens)

    @classmethod
    def from_args(cls, args, rest: list[str] = ()) -> list[str]:
        out: list[str] = []
        for name, _, _ in (*cls.REQUIRED, *cls.OPTIONAL):
            value = getattr(args, name)
            if value is not None:
                out += [f"--{name}", str(value)]
        return [*out, *rest]


@dataclass
class Step:
    name: str
    command: list[str]
    timeout_s: float = 1800.0
    env: tuple[tuple[str, str], ...] = ()


@dataclass
class StepResult:
    name: str
    ok: bool
    seconds: float
    error: str = ""


def run_process(step: Step, log: Path) -> StepResult:
    import os
    started = time.monotonic()
    try:
        with open(log, "w", encoding="utf8") as handle:
            handle.write("$ " + " ".join(step.command) + "\n")
            handle.flush()
            done = subprocess.run(step.command, stdout=handle, stderr=subprocess.STDOUT, timeout=step.timeout_s, check=False, env={**os.environ, **dict(step.env)})
        error = "" if done.returncode == 0 else f"exit code {done.returncode}"
    except subprocess.TimeoutExpired:
        error = f"timed out after {step.timeout_s:.0f} s"
    return StepResult(step.name, not error, time.monotonic() - started, error)


def default_scenarios(*, warmup: int, ceilings: str) -> list[Scenario]:
    """The open questions in one list. The first scenario (native, eager) is the baseline every other is compared with."""
    int8 = ("--param", "adapter.base_precision=int8")
    graph = ("--set", "trainer.graph=true")
    lora = ("--param", "adapter.lora_compute=activation")
    geglu = ("--param", "trainer.fused_geglu=true")
    measured = ("--param", "trainer.checkpoint_by=measured", "--param", "trainer.tune_checkpointing=true")
    # Options the scenarios vary are stated in every scenario (later arguments win), so options the user put in the shared arguments cannot change a baseline.
    neutral = ("--param", "adapter.lora_compute=float32", "--param", "trainer.fused_geglu=false")
    window = ("--set", f"perf.warmup_steps={warmup}", "--set", f"perf.ceilings={ceilings}", "--set", "perf.provenance=true", *neutral)  # one measurement of the ceilings for every scenario
    S = lambda name, *args, env=(): Scenario(name, (*window, *args), tuple(env))  # noqa: E731
    return [S("native eager"), S("native graph", *graph), S("int8 eager", *int8), S("int8 graph", *int8, *graph),
            S("int8 dx-int8 eager", *int8, "--param", "adapter.int8_grad=int8"), S("int8 dx-int8 graph", *int8, "--param", "adapter.int8_grad=int8", *graph),
            S("native graph measured-checkpoints", *graph, *measured), S("int8 graph measured-checkpoints", *int8, *graph, *measured),
            S("native graph attention-math_bf16", *graph, "--param", "trainer.graph_attention=math_bf16"),
            S("native graph attention-chunked_bf16", *graph, "--param", "trainer.graph_attention=chunked_bf16"),
            S("int8 graph attention-chunked_bf16", *int8, *graph, "--param", "trainer.graph_attention=chunked_bf16"),
            S("native eager lora-activation", *lora), S("native graph attention-math_bf16 lora-activation", *graph, "--param", "trainer.graph_attention=math_bf16", *lora),
            S("int8 graph attention-math_bf16 lora-activation", *int8, *graph, "--param", "trainer.graph_attention=math_bf16", *lora),
            S("int8 graph attention-math_bf16 lora-activation fused-geglu", *int8, *graph, "--param", "trainer.graph_attention=math_bf16", *lora, *geglu),
            S("native graph attention-math_bf16 lora-activation fused-geglu", *graph, "--param", "trainer.graph_attention=math_bf16", *lora, *geglu),
            S("native eager lora-activation fused-geglu", *lora, *geglu),
            S("native expandable-segments", env=(("PYTORCH_ALLOC_CONF", "expandable_segments:True"),)),
            S("native no-expandable-segments", env=(("PYTORCH_ALLOC_CONF", "expandable_segments:False"),))]


class HwCheck:
    """``train_args`` are the arguments of ``python -m next.run sdxl-lora`` that describe the model, dataset and size (everything but steps and --perf)."""

    def __init__(self, out: Path, train_args: list[str], *, device: str = "xpu", steps: int = 28, warmup: int = 10, python: str = sys.executable,
                 run_step: Callable[[Step, Path], StepResult] = run_process, say: Callable[[str], None] = print, only: set[str] | None = None,
                 bench_args: tuple[str, ...] = (), scenarios: list[str] | None = None):
        self.out, self.train_args, self.device, self.steps, self.warmup = out, train_args, device, steps, warmup
        self.python, self.run_step, self.say, self.only, self.bench_args = python, run_step, say, only, bench_args
        self.scenario_names = scenarios
        self.results: list[StepResult] = []

    def wanted(self, name: str) -> bool:
        return self.only is None or name in self.only

    def steps_list(self) -> list[Step]:
        py, out = self.python, self.out
        m = [py, "-m", "next.perf"]
        return [Step("hardware", [*m, "hardware", "--out", str(out / "hardware")], 300),
                Step("allocator expandable", [*m, "allocator", "--device", self.device], 300, (("PYTORCH_ALLOC_CONF", "expandable_segments:True"),)),
                Step("allocator plain", [*m, "allocator", "--device", self.device], 300, (("PYTORCH_ALLOC_CONF", "expandable_segments:False"),)),
                Step("events", [*m, "events", "--device", self.device, "--out", str(out / "events")], 300),
                Step("bench", [*m, "bench", "--device", self.device, "--out", str(out / "bench"), *self.bench_args], 1800)]

    def run(self) -> Path:
        self.out.mkdir(parents=True, exist_ok=True)
        logs = self.out / "logs"
        logs.mkdir(exist_ok=True)
        for step in self.steps_list():
            group = step.name.split()[0]
            if not self.wanted(group):
                continue
            self.say(f"hwcheck: {step.name} ...")
            result = self.run_step(step, logs / f"{step.name.replace(' ', '_')}.log")
            self.results.append(result)
            self.say(f"    {'ok' if result.ok else 'FAILED: ' + result.error} in {result.seconds:.0f} s")
        if self.wanted("matrix"):
            self._matrix()
            if self.wanted("verdict"):
                self._verdict(logs)
        index = self._index()
        self._zip()
        return index

    def train_command(self) -> list[str]:
        return [self.python, "-m", "next.run", "sdxl-lora", *self.train_args, "--steps", str(self.steps)]

    def scenarios(self) -> list[Scenario]:
        """All of them, or those whose name or folder name was asked for (the first, the baseline, is always kept)."""
        every = default_scenarios(warmup=self.warmup, ceilings=str(self.out / "ceilings.json"))
        if not self.scenario_names:
            return every
        wanted = {n.lower() for n in self.scenario_names}
        unknown = wanted - {s.name.lower() for s in every} - {s.slug.lower() for s in every}
        if unknown:
            raise ValueError(f"unknown scenario(s) {sorted(unknown)}; choose from: " + ", ".join(s.name for s in every))
        return [s for i, s in enumerate(every) if i == 0 or s.name.lower() in wanted or s.slug.lower() in wanted]

    def _matrix(self) -> None:
        command = self.train_command()
        spec = MatrixSpec(command, self.scenarios(), timeout_s=3600.0)
        started = time.monotonic()
        matrix = Matrix(spec, self.out / "matrix", SubprocessRunner(command, spec.timeout_s), resume=True, say=self.say)
        matrix.run()
        failed = [o for o in matrix.outcomes if not o.ok]
        self.results.append(StepResult("matrix", not failed, time.monotonic() - started, f"{len(failed)} of {len(matrix.outcomes)} scenarios failed" if failed else ""))

    def _verdict(self, logs: Path) -> None:
        """The acceptance lab judges the matrix folders (run as a command: the perf layer does not import it)."""
        step = Step("verdict", [self.python, "-m", "next.accept", "judge", str(self.out / "matrix"), "speed", "allocator", "--out", str(self.out / "verdict"),
                                "--warmup", str(self.warmup)], 300)
        self.say("hwcheck: verdict ...")
        result = self.run_step(step, logs / "verdict.log")
        self.results.append(result)
        self.say(f"    {'ok' if result.ok else 'FAILED: ' + result.error} in {result.seconds:.0f} s")

    def _index(self) -> Path:
        rows = ["# Hardware check bundle", "", "| step | result | seconds |", "|---|---|---:|"]
        rows += [f"| {r.name} | {'ok' if r.ok else 'FAILED: ' + r.error} | {r.seconds:.0f} |" for r in self.results]
        rows += ["", "Send `bundle.zip` (text files only), or these files: `hardware/hardware.md`, `bench/bench.md`, `matrix/summary.md`, the `report.md` inside each `matrix/<scenario>/` folder,",
                 "`verdict/verdict.md` (the acceptance lab's pass/fail per check) and `logs/` for anything that failed. The allocator steps print one JSON line each (in `logs/`): equal numbers between the two mean the allocator setting changes nothing."]
        path = self.out / "README.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf8")
        return path

    def _zip(self) -> Path:
        return pack(self.out)
