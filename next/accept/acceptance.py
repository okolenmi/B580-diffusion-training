"""Running suites: the scenarios of the chosen suites through the perf lab's matrix runner (each in its own process), then the judgement."""

from __future__ import annotations

import sys
from pathlib import Path
from collections.abc import Callable

from ..perf.bundle import pack
from ..perf.matrix import Matrix, MatrixSpec, Runner, Scenario, SubprocessRunner
from .suites import Suite, SuiteBook
from .verdict import Judge, SuiteVerdict, write_verdict


class Acceptance:
    """``train_args`` are the arguments of ``python -m next.run sdxl-lora`` that describe the model, dataset and size (not steps, not --perf)."""

    def __init__(self, suites: list[Suite], out: Path, train_args: list[str], *, steps: int = 28, python: str = sys.executable,
                 runner_for: Callable[[list[str]], Runner] | None = None, say: Callable[[str], None] = print):
        self.suites, self.out, self.say = suites, out, say
        self.command = [python, "-m", "next.run", "sdxl-lora", *train_args, "--steps", str(steps)]
        self.runner = (runner_for or (lambda command: SubprocessRunner(command)))(self.command)

    def scenarios(self) -> list[Scenario]:
        seen: dict[str, Scenario] = {}
        for suite in self.suites:
            for scenario in suite.scenarios:
                seen.setdefault(scenario.name, scenario)
        return list(seen.values())

    def run(self) -> Path:
        spec = MatrixSpec(self.command, self.scenarios())
        Matrix(spec, self.out / "matrix", self.runner, resume=True, say=self.say).run()
        verdict = self.judge(self.out / "matrix", self.out)
        self.say(f"text bundle (logs, verdict, reports; no adapters or traces): {pack(self.out)}")
        return verdict

    def judge(self, root: Path, out: Path) -> Path:
        judge = Judge(root)
        verdicts: list[SuiteVerdict] = [judge.judge(suite) for suite in self.suites]
        return write_verdict(verdicts, out)


def choose(names: list[str] | None, book: SuiteBook) -> list[Suite]:
    suites = book.all()
    unknown = [n for n in names or [] if n not in suites]
    if unknown:
        raise SystemExit(f"unknown suite {unknown[0]!r}; choose from {', '.join(suites)}")
    return [suites[n] for n in names] if names else list(suites.values())
