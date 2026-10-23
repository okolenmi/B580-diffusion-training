"""Collecting results into one verdict and writing it as markdown the owner reads in a minute."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .artifacts import RunArtifacts
from .checks import Result, Status
from .suites import Suite


@dataclass
class SuiteVerdict:
    suite: Suite
    results: list[Result] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)  # scenarios with no folder

    def count(self, status: Status) -> int:
        return sum(r.status is status for r in self.results)

    @property
    def status(self) -> Status:
        if self.count(Status.FAIL):
            return Status.FAIL
        return Status.LOOK if self.count(Status.LOOK) else Status.PASS

    def to_json(self) -> dict:
        return {"suite": self.suite.name, "item": self.suite.item, "status": self.status.value, "missing": self.missing,
                "results": [{"check": r.name, "status": r.status.value, "message": r.message, "reason": r.reason} for r in self.results]}

    def markdown(self) -> str:
        s = self.suite
        out = [f"## {s.title} (TESTING {s.item}): {self.status.value}", "",
               f"{self.count(Status.PASS)} passed, {self.count(Status.FAIL)} failed, {self.count(Status.LOOK)} to look at, {self.count(Status.SKIP)} skipped."]
        if self.missing:
            out += ["", "No folder for: " + ", ".join(self.missing) + " (not run, or not in this bundle)."]
        out += ["", "| check | result | detail |", "|---|---|---|"]
        order = {Status.FAIL: 0, Status.LOOK: 1, Status.SKIP: 2, Status.PASS: 3}
        for r in sorted(self.results, key=lambda r: order[r.status]):
            detail = r.message.replace("\n", " / ").replace("|", "/")
            out.append(f"| {r.name} | {r.status.value} | {detail}{f' (from: {r.reason})' if r.reason and r.status is Status.FAIL else ''} |")
        return "\n".join(out)


class Judge:
    """Reads each scenario's folder under ``root`` (named by the scenario's slug) and evaluates a suite's checks."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def judge(self, suite: Suite) -> SuiteVerdict:
        verdict = SuiteVerdict(suite)
        scenarios = {s.name: s for s in suite.scenarios}
        runs: dict[str, RunArtifacts | None] = {}
        for name, scenario in scenarios.items():
            folder = self.root / scenario.slug
            runs[name] = RunArtifacts.read(folder) if folder.is_dir() else None
            if runs[name] is None:
                verdict.missing.append(name)
        for check in suite.checks:
            needed = [runs.get(n) for n in check.scenarios]
            if any(r is None for r in needed):
                absent = [n for n, r in zip(check.scenarios, needed) if r is None]
                verdict.results.append(Result(check.name, Status.SKIP, "no run for " + ", ".join(absent), check.reason))
                continue
            try:
                verdict.results.append(check.evaluate(needed))
            except Exception as error:  # noqa: BLE001 - a broken check is a result, not a crash of the judgement
                verdict.results.append(Result(check.name, Status.FAIL, f"the check itself failed: {type(error).__name__}: {error}", check.reason))
        return verdict


def write_verdict(verdicts: list[SuiteVerdict], folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    total = {s: sum(v.count(s) for v in verdicts) for s in Status}
    out = ["# Acceptance verdict", "", f"{total[Status.PASS]} passed, {total[Status.FAIL]} failed, {total[Status.LOOK]} for your eyes, {total[Status.SKIP]} skipped.", "",
           "| suite | TESTING | result |", "|---|---|---|"]
    out += [f"| {v.suite.title} | {v.suite.item} | {v.status.value} |" for v in verdicts]
    out.append("")
    for v in verdicts:
        out += [v.markdown(), ""]
    looks = [(v.suite.title, r) for v in verdicts for r in v.results if r.status is Status.LOOK]
    if looks:
        out += ["## Needs your eyes", ""] + [f"- [ ] {title}: {r.name}. {r.message}" for title, r in looks]
    (folder / "verdict.json").write_text(json.dumps([v.to_json() for v in verdicts], indent=1))
    path = folder / "verdict.md"
    path.write_text("\n".join(out) + "\n", encoding="utf8")
    return path
