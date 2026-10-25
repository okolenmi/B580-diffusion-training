"""A matrix of runs: named scenarios, each in its own process (a fresh allocator and primitive cache), collected into one summary.

Scenario file (TOML)::

    [base]
    command = ["python", "-m", "next.run", "sdxl-lora", "--models", "M", "--datasets", "D", "--file", "x.safetensors", "--dataset", "d", "--output", "out", "--steps", "14"]
    timeout_s = 1800

    [axes.precision]          # every combination of the axes is a scenario
    native = []
    int8 = ["--param", "adapter.base_precision=int8"]
    [axes.graph]
    eager = []
    graph = ["--set", "trainer.graph=true"]

    [[scenario]]              # or named scenarios by hand
    name = "int8 dx-int8"
    args = ["--param", "adapter.base_precision=int8", "--param", "adapter.int8_grad=int8"]
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .comparison import Comparison


@dataclass(frozen=True)
class Scenario:
    name: str
    args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()

    @property
    def slug(self) -> str:
        return "".join(c if c.isalnum() or c in "-_=" else "_" for c in self.name.replace(" ", "_"))


@dataclass
class Outcome:
    scenario: Scenario
    ok: bool
    seconds: float
    folder: Path
    log: Path | None = None
    error: str = ""
    report: dict | None = None


@dataclass
class MatrixSpec:
    command: list[str]
    scenarios: list[Scenario]
    timeout_s: float = 3600.0

    @classmethod
    def load(cls, path: str | Path) -> "MatrixSpec":
        data = tomllib.loads(Path(path).read_text(encoding="utf8"))
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "MatrixSpec":
        base = data.get("base") or {}
        command = list(base.get("command") or [])
        if not command:
            raise ValueError("the scenario file needs base.command")
        scenarios: list[Scenario] = []
        axes = data.get("axes") or {}
        if axes:
            names = list(axes)
            for combo in itertools.product(*[list(axes[n].items()) for n in names]):
                label = ",".join(f"{n}={value}" for n, (value, _) in zip(names, combo))
                scenarios.append(Scenario(label, tuple(a for _, args in combo for a in args)))
        for entry in data.get("scenario") or []:
            scenarios.append(Scenario(entry["name"], tuple(entry.get("args", ())), tuple((entry.get("env") or {}).items())))
        if not scenarios:
            raise ValueError("the scenario file defines no scenarios (axes or [[scenario]])")
        slugs = [s.slug for s in scenarios]
        if len(set(slugs)) != len(slugs):
            raise ValueError("scenario names must differ once spaces and symbols are replaced")
        return cls(command, scenarios, float(base.get("timeout_s", 3600.0)))


class Runner(Protocol):
    def run(self, scenario: Scenario, folder: Path) -> Outcome: ...


class SubprocessRunner:
    """Runs ``command + scenario.args + ["--perf", folder]``; stdout and stderr go to ``folder/run.log``."""

    def __init__(self, command: list[str], timeout_s: float = 3600.0):
        self.command, self.timeout_s = command, timeout_s

    def run(self, scenario: Scenario, folder: Path) -> Outcome:
        folder.mkdir(parents=True, exist_ok=True)
        log = folder / "run.log"
        started = time.monotonic()
        full = [*self.command, *(a.replace("{folder}", str(folder)) for a in scenario.args), "--perf", str(folder)]  # {folder}: per-scenario output paths
        try:
            with open(log, "w", encoding="utf8") as handle:
                handle.write("$ " + " ".join(full) + "\n")
                handle.flush()
                done = subprocess.run(full, stdout=handle, stderr=subprocess.STDOUT, env={**os.environ, **dict(scenario.env)}, timeout=self.timeout_s, check=False)
            error = "" if done.returncode == 0 else f"exit code {done.returncode}"
        except subprocess.TimeoutExpired:
            error = f"timed out after {self.timeout_s:.0f} s"
        return Outcome(scenario, not error, time.monotonic() - started, folder, log, error)


def load_report(folder: Path) -> dict | None:
    path = folder / "report.json"
    return json.loads(path.read_text()) if path.exists() else None


@dataclass
class Matrix:
    spec: MatrixSpec
    folder: Path
    runner: Runner
    resume: bool = False
    say: callable = print
    outcomes: list[Outcome] = field(default_factory=list)

    def run(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        for i, scenario in enumerate(self.spec.scenarios, 1):
            where = self.folder / scenario.slug
            report = load_report(where) if self.resume else None
            if report is not None:
                self.say(f"[{i}/{len(self.spec.scenarios)}] {scenario.name}: report exists, kept")
                self.outcomes.append(Outcome(scenario, True, 0.0, where, where / "run.log", report=report))
                continue
            self.say(f"[{i}/{len(self.spec.scenarios)}] {scenario.name} ...")
            outcome = self.runner.run(scenario, where)
            outcome.report = load_report(where)
            if outcome.ok and outcome.report is None:
                outcome.ok, outcome.error = False, "the run finished but wrote no report (it ended before the profiled window?)"
            self.say(f"    {'ok' if outcome.ok else 'FAILED: ' + outcome.error} in {outcome.seconds:.0f} s")
            self.outcomes.append(outcome)
        path = self.folder / "summary.md"
        path.write_text(self.summary(), encoding="utf8")
        return path

    def summary(self) -> str:
        done = [o for o in self.outcomes if o.report]
        out = ["# Matrix summary", "", "| scenario | status | wall ms | eager ms | replayed ms | device ms | utilisation | power W | peak MB | top finding |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for o in self.outcomes:
            if not o.report:
                out.append(f"| {o.scenario.name} | FAILED: {o.error} | | | | | | | | log: {o.log} |")
                continue
            b = o.report["budget"]
            finding = o.report["findings"][0]["title"] if o.report["findings"] else "-"
            f = lambda v, s=".0f": "-" if v is None else format(v, s)  # noqa: E731
            out.append(f"| {o.scenario.name} | ok | {f(b['wall_ms'])} | {f(b['eager_ms'])} | {f(b['replayed_ms'])} | {f(b['device_ms'])} | "
                       f"{'-' if b['utilisation'] is None else format(b['utilisation'], '.0%')} | {f(o.report.get('meta', {}).get('power_w'))} | {f(b.get('peak_mb'))} | {finding} |")
        if len(done) > 1:
            first = done[0]
            out += ["", f"Every other scenario below is compared with the first, `{first.scenario.name}`.", ""]
            for o in done[1:]:
                out.append(Comparison(first.report, o.report, first.scenario.name, o.scenario.name).markdown())
        return "\n".join(out) + "\n"
