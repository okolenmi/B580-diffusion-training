"""One command for the owner's pending hardware runs: every step in its own process (fresh allocator, nothing held by this one), one folder,
one zip of text files to send back. A step that fails is recorded and the rest still run."""

from __future__ import annotations

import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .matrix import Matrix, MatrixSpec, Scenario, SubprocessRunner

LOG_CAP = 200_000
TEXT_SUFFIXES = (".md", ".json", ".log")


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
    measured = ("--param", "trainer.checkpoint_by=measured", "--param", "trainer.tune_checkpointing=true")
    window = ("--set", f"perf.warmup_steps={warmup}", "--set", f"perf.ceilings={ceilings}")  # one measurement of the ceilings for every scenario
    S = lambda name, *args, env=(): Scenario(name, (*window, *args), tuple(env))  # noqa: E731
    return [S("native eager"), S("native graph", *graph), S("int8 eager", *int8), S("int8 graph", *int8, *graph),
            S("int8 dx-int8 eager", *int8, "--param", "adapter.int8_grad=int8"), S("int8 dx-int8 graph", *int8, "--param", "adapter.int8_grad=int8", *graph),
            S("native graph measured-checkpoints", *graph, *measured), S("int8 graph measured-checkpoints", *int8, *graph, *measured),
            S("native expandable-segments", env=(("PYTORCH_ALLOC_CONF", "expandable_segments:True"),)),
            S("native no-expandable-segments", env=(("PYTORCH_ALLOC_CONF", "expandable_segments:False"),))]


class HwCheck:
    """``train_args`` are the arguments of ``python -m next.run train`` that describe the model, dataset and size (everything but steps and --perf)."""

    def __init__(self, out: Path, train_args: list[str], *, device: str = "xpu", steps: int = 28, warmup: int = 20, python: str = sys.executable,
                 run_step: Callable[[Step, Path], StepResult] = run_process, say: Callable[[str], None] = print, only: set[str] | None = None,
                 bench_args: tuple[str, ...] = ()):
        self.out, self.train_args, self.device, self.steps, self.warmup = out, train_args, device, steps, warmup
        self.python, self.run_step, self.say, self.only, self.bench_args = python, run_step, say, only, bench_args
        self.results: list[StepResult] = []

    def wanted(self, name: str) -> bool:
        return self.only is None or name in self.only

    def steps_list(self) -> list[Step]:
        py, out = self.python, self.out
        m = [py, "-m", "next.perf"]
        return [Step("hardware", [*m, "hardware", "--out", str(out / "hardware")], 300),
                Step("allocator expandable", [*m, "allocator", "--device", self.device], 300, (("PYTORCH_ALLOC_CONF", "expandable_segments:True"),)),
                Step("allocator plain", [*m, "allocator", "--device", self.device], 300, (("PYTORCH_ALLOC_CONF", "expandable_segments:False"),)),
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
        index = self._index()
        self._zip()
        return index

    def _matrix(self) -> None:
        command = [self.python, "-m", "next.run", "train", *self.train_args, "--steps", str(self.steps)]
        spec = MatrixSpec(command, default_scenarios(warmup=self.warmup, ceilings=str(self.out / 'ceilings.json')), timeout_s=3600.0)
        started = time.monotonic()
        matrix = Matrix(spec, self.out / "matrix", SubprocessRunner(command, spec.timeout_s), resume=True, say=self.say)
        matrix.run()
        failed = [o for o in matrix.outcomes if not o.ok]
        self.results.append(StepResult("matrix", not failed, time.monotonic() - started, f"{len(failed)} of {len(matrix.outcomes)} scenarios failed" if failed else ""))

    def _index(self) -> Path:
        rows = ["# Hardware check bundle", "", "| step | result | seconds |", "|---|---|---:|"]
        rows += [f"| {r.name} | {'ok' if r.ok else 'FAILED: ' + r.error} | {r.seconds:.0f} |" for r in self.results]
        rows += ["", "Send `bundle.zip` (text files only), or these files: `hardware/hardware.md`, `bench/bench.md`, `matrix/summary.md`, the `report.md` inside each `matrix/<scenario>/` folder,",
                 "and `logs/` for anything that failed. The allocator steps print one JSON line each (in `logs/`): equal numbers between the two mean the allocator setting changes nothing."]
        path = self.out / "README.md"
        path.write_text("\n".join(rows) + "\n", encoding="utf8")
        return path

    def _zip(self) -> Path:
        path = self.out / "bundle.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(self.out.rglob("*")):
                if not file.is_file() or file == path or file.suffix not in TEXT_SUFFIXES or file.name == "trace.json":
                    continue
                data = file.read_bytes()
                if len(data) > LOG_CAP and file.suffix == ".log":
                    data = data[:LOG_CAP // 4] + b"\n... (middle cut) ...\n" + data[-LOG_CAP * 3 // 4:]
                archive.writestr(str(file.relative_to(self.out)), data)
        return path
