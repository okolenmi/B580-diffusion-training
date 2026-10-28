"""What every numeric check returns, and how a run of them is reported.

A check is *exact* when the two things compared must be the same bits (a copy, an offload and reload, a repeated launch): any
difference is a FAIL. It is *measured* when two valid computations are allowed to differ (a rounding path, an attention
variant): the numbers are shown, and only a difference far past what rounding explains is a WARN.
"""

from __future__ import annotations

import json
import traceback
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PASS, FAIL, WARN, INFO, SKIP = "PASS", "FAIL", "WARN", "INFO", "SKIP"
ORDER = {FAIL: 0, WARN: 1, INFO: 2, PASS: 3, SKIP: 4}


@dataclass
class Finding:
    check: str
    name: str
    status: str
    detail: str
    numbers: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"{self.status:<4} {self.check}: {self.name} -- {self.detail}"


class NumericCheck(ABC):
    """One question about the numbers. ``needs_model`` checks run on the subject (a real checkpoint or a tiny stand-in)."""

    name = ""
    purpose = ""
    needs_model = False

    @abstractmethod
    def run(self, ctx) -> list[Finding]: ...

    # helpers so a check reads as the question it asks
    def exact(self, name: str, differing: int, total: int, unit: str = "values") -> Finding:
        status = PASS if differing == 0 else FAIL
        return Finding(self.name, name, status, f"{differing} of {total} {unit} differ", {"differing": differing, "total": total})

    def measured(self, name: str, detail: str, numbers: dict, warn_if: bool = False) -> Finding:
        return Finding(self.name, name, WARN if warn_if else INFO, detail, numbers)

    def skip(self, name: str, why: str) -> Finding:
        return Finding(self.name, name, SKIP, why)


class NumericsRunner:
    """Runs the checks one after another; a check that raises is a FAIL with its traceback, and the rest still run."""

    def __init__(self, checks: list[NumericCheck], ctx, say: Callable[[str], None] = print, cleanup: Callable[[], None] = lambda: None):
        self.checks, self.ctx, self.say, self.cleanup = checks, ctx, say, cleanup

    def run(self) -> list[Finding]:
        findings: list[Finding] = []
        for check in self.checks:
            self.say(f"[{check.name}] {check.purpose}")
            try:
                got = check.run(self.ctx)
            except Exception as error:  # noqa: BLE001 - a crashing check is a finding, not the end of the run
                tail = "".join(traceback.format_exception_only(type(error), error)).strip()
                got = [Finding(check.name, "crashed", FAIL, tail, {"traceback": traceback.format_exc()})]
            for finding in got:
                self.say("  " + finding.line())
            findings.extend(got)
            self.cleanup()
        return findings


def counts(findings: list[Finding]) -> dict[str, int]:
    out = {status: 0 for status in ORDER}
    for finding in findings:
        out[finding.status] += 1
    return out


def markdown(findings: list[Finding], header: str) -> str:
    tally = counts(findings)
    lines = ["# Numerics report", "", header, "", ", ".join(f"{n} {s}" for s, n in tally.items() if n), "",
             "PASS/FAIL are for checks that must be bit-exact. INFO lines are measurements of two valid computations that may differ; "
             "WARN means the difference is past what rounding explains.", "",
             "| status | check | what | result |", "|---|---|---|---|"]
    for f in sorted(findings, key=lambda f: ORDER[f.status]):
        lines.append(f"| {f.status} | {f.check} | {f.name} | {f.detail.replace('|', '/')} |")
    for f in findings:
        if "traceback" in f.numbers:
            lines += ["", f"### {f.check} crashed", "```", f.numbers["traceback"].rstrip(), "```"]
    return "\n".join(lines) + "\n"


def write(findings: list[Finding], folder: str | Path, header: str) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "numerics.json").write_text(json.dumps([f.__dict__ for f in findings], indent=1, default=str), encoding="utf8")
    path = folder / "numerics.md"
    path.write_text(markdown(findings, header), encoding="utf8")
    return path
