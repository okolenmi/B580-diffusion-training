"""Judgements over run artifacts. A check reads artifacts and returns a result; it never runs anything and never imports the trainer."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .artifacts import RunArtifacts


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    LOOK = "LOOK"   # decided by eyes: the message names what to open
    SKIP = "SKIP"   # the data the check needs is not there


@dataclass
class Result:
    name: str
    status: Status
    message: str
    reason: str = ""  # where the threshold comes from


@dataclass
class Check:
    """One judgement. ``scenarios`` are the names it reads (one for a run check, candidate then baseline for a pair check)."""

    name: str
    scenarios: tuple[str, ...]
    reason: str = ""

    def evaluate(self, runs: list[RunArtifacts]) -> Result:
        raise NotImplementedError

    def result(self, status: Status, message: str) -> Result:
        return Result(self.name, status, message, self.reason)

    def _passes(self, ok: bool, good, bad) -> Result:
        """``good`` and ``bad`` are text, or functions returning text (so the one not used is never built)."""
        message = good if ok else bad
        return self.result(Status.PASS if ok else Status.FAIL, message() if callable(message) else message)


def _median_ms(run: RunArtifacts, skip: int) -> float | None:
    values = [s.ms for s in run.settled_steps(skip)]
    return statistics.median(values) if values else None


class RunFinished(Check):
    def evaluate(self, runs):
        run = runs[0]
        if run.finished:
            return self.result(Status.PASS, "the run finished")
        return self.result(Status.FAIL, f"the run did not finish: {run.failed_with or 'no final line'}\n{run.tail()}")


class LinesPresent(Check):
    def __init__(self, name, scenarios, pattern: str, at_least: int = 1, reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.pattern, self.at_least = re.compile(pattern), at_least

    def evaluate(self, runs):
        found = [l for l in runs[0].lines if self.pattern.search(l)]
        return self._passes(len(found) >= self.at_least, f"{len(found)} line(s) match {self.pattern.pattern!r}: {found[0][:140]}" if found else "",
                            f"{len(found)} line(s) match {self.pattern.pattern!r}, expected at least {self.at_least}")


class LinesAbsent(Check):
    def __init__(self, name, scenarios, pattern: str, reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.pattern = re.compile(pattern)

    def evaluate(self, runs):
        found = [l for l in runs[0].lines if self.pattern.search(l)]
        first = found[0][:160] if found else ""
        return self._passes(not found, f"no line matches {self.pattern.pattern!r}", f"{len(found)} line(s) match {self.pattern.pattern!r}: {first}")


class LossFinite(Check):
    def evaluate(self, runs):
        steps = runs[0].steps
        if not steps:
            return self.result(Status.SKIP, "no step lines in the log")
        bad = [s.step for s in steps if not math.isfinite(s.loss)]
        return self._passes(not bad, f"{len(steps)} losses, all finite", f"non-finite loss at step(s) {bad[:5]}")


class PeakBelow(Check):
    def __init__(self, name, scenarios, mb: float, field: str = "peak_mb", reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.mb, self.field = mb, field

    def evaluate(self, runs):
        values = [getattr(s, self.field) for s in runs[0].steps if getattr(s, self.field) is not None]
        if not values:
            return self.result(Status.SKIP, f"no {self.field} in the step lines")
        return self._passes(max(values) <= self.mb, f"{self.field} peaked at {max(values)} MB (limit {self.mb:.0f})", f"{self.field} reached {max(values)} MB, over {self.mb:.0f}")


class PeakAtLeast(Check):
    def __init__(self, name, scenarios, mb: float, field: str = "device_mb", reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.mb, self.field = mb, field

    def evaluate(self, runs):
        values = [getattr(s, self.field) for s in runs[0].steps if getattr(s, self.field) is not None]
        if not values:
            return self.result(Status.SKIP, f"no {self.field} in the step lines")
        return self._passes(max(values) >= self.mb, f"{self.field} reached {max(values)} MB (wanted {self.mb:.0f} or more)", f"{self.field} only reached {max(values)} MB, wanted {self.mb:.0f}")


class PeakLower(Check):
    """Candidate's peak at least ``by_mb`` under the baseline's."""

    def __init__(self, name, scenarios, by_mb: float, field: str = "peak_mb", reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.by_mb, self.field = by_mb, field

    def evaluate(self, runs):
        peaks = [max((getattr(s, self.field) for s in r.steps if getattr(s, self.field) is not None), default=None) for r in runs]
        if None in peaks:
            return self.result(Status.SKIP, f"no {self.field} in one of the logs")
        saved = peaks[1] - peaks[0]
        return self._passes(saved >= self.by_mb, f"{self.field} {peaks[0]} against {peaks[1]} MB: {saved} MB lower (wanted {self.by_mb:.0f})",
                            f"{self.field} {peaks[0]} against {peaks[1]} MB: only {saved} MB lower, wanted {self.by_mb:.0f}")


class StepTimeWithin(Check):
    """Candidate's median settled step at most ``ratio`` times the baseline's."""

    def __init__(self, name, scenarios, ratio: float, skip: int = 0, reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.ratio, self.skip = ratio, skip

    def evaluate(self, runs):
        times = [_median_ms(r, self.skip) for r in runs]
        if None in times:
            return self.result(Status.SKIP, "a log has no settled step lines")
        ratio = times[0] / times[1]
        return self._passes(ratio <= self.ratio, f"median step {times[0]:.0f} ms against {times[1]:.0f} ms: x{ratio:.2f} (limit x{self.ratio:.2f})",
                            f"median step {times[0]:.0f} ms against {times[1]:.0f} ms: x{ratio:.2f}, over x{self.ratio:.2f}")


class StepTimeBelow(Check):
    def __init__(self, name, scenarios, ms: float, skip: int = 0, reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.ms, self.skip = ms, skip

    def evaluate(self, runs):
        median = _median_ms(runs[0], self.skip)
        if median is None:
            return self.result(Status.SKIP, "no settled step lines")
        return self._passes(median <= self.ms, f"median step {median:.0f} ms (limit {self.ms:.0f})", f"median step {median:.0f} ms, over {self.ms:.0f}")


class LossCurveClose(Check):
    """The last ``tail`` losses of the candidate stay within ``rel`` of the baseline's mean over the same steps."""

    def __init__(self, name, scenarios, rel: float, tail: int = 10, reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.rel, self.tail = rel, tail

    def evaluate(self, runs):
        curves = [[s.loss for s in r.steps][-self.tail:] for r in runs]
        if not all(curves):
            return self.result(Status.SKIP, "a log has no step lines")
        a, b = statistics.fmean(curves[0]), statistics.fmean(curves[1])
        gap = abs(a - b) / max(abs(b), 1e-12)
        return self._passes(gap <= self.rel, f"mean loss over the last {self.tail} steps {a:.4f} against {b:.4f} ({gap:.1%}, limit {self.rel:.0%})",
                            f"mean loss over the last {self.tail} steps {a:.4f} against {b:.4f} ({gap:.1%}), over {self.rel:.0%}")


class GraphDecided(Check):
    """Every captured shape got a replay-against-eager line; ``expect`` says whether the graphs should all be kept, all dropped, or either."""

    def __init__(self, name, scenarios, expect: str = "any", reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.expect = expect

    def evaluate(self, runs):
        decisions = runs[0].graph_decisions
        if not decisions:
            return self.result(Status.FAIL, "no '[graph] shape ...' decision line (no shape reached its timed replays in this run?)")
        dropped = sum(d.dropped for d in decisions)
        text = "; ".join(f"{d.shape}: replay {d.replay_ms} ms against {d.eager_ms} ms eager{' dropped' if d.dropped else ' kept'}" for d in decisions[:4])
        wanted = {"kept": dropped == 0, "dropped": dropped == len(decisions), "any": True}[self.expect]
        return self._passes(wanted, text, f"expected the graphs to be {self.expect}: {text}")


class ReportFinding(Check):
    """The perf report's findings do not contain a title (``absent=True``) or do contain one."""

    def __init__(self, name, scenarios, title: str, absent: bool = True, reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.title, self.absent = title, absent

    def evaluate(self, runs):
        report = runs[0].report
        if report is None:
            return self.result(Status.SKIP, "no perf report")
        has = any(self.title in f["title"] for f in report["findings"])
        return self._passes(has != self.absent, f"finding {self.title!r} is {'present' if has else 'absent'}", f"finding {self.title!r} is {'present' if has else 'absent'}")


class AdaptersIdentical(Check):
    """Both runs saved adapter files (``pattern`` under each folder) with exactly equal tensors."""

    def __init__(self, name, scenarios, pattern: str = "out/**/*.safetensors", reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.pattern = pattern

    def evaluate(self, runs):
        from safetensors.torch import load_file
        files = [r.files(self.pattern) for r in runs]
        if not all(files):
            return self.result(Status.SKIP, "a run saved no adapter file")
        a, b = load_file(str(files[0][-1])), load_file(str(files[1][-1]))
        if a.keys() != b.keys():
            return self.result(Status.FAIL, "the adapter files hold different keys")
        different = [k for k in a if not (a[k].shape == b[k].shape and bool((a[k] == b[k]).all()))]
        return self._passes(not different, f"{len(a)} tensors, all exactly equal", lambda: f"{len(different)} of {len(a)} tensors differ, e.g. {different[0]}")


@dataclass
class LookAt(Check):
    """A judgement only eyes can make: always LOOK, naming what to open and what to look for."""

    instruction: str = ""
    patterns: tuple[str, ...] = ()

    def evaluate(self, runs):
        found = [str(p.relative_to(runs[0].folder)) for pattern in self.patterns for p in runs[0].files(pattern)]
        where = ", ".join(found) if found else "(no matching file was produced)"
        return self.result(Status.LOOK if found else Status.SKIP, f"{self.instruction} Open: {where}")


class NumbersShown(Check):
    """A diagnostic with no threshold: shows one field's peak in each run and asks for a look."""

    def __init__(self, name, scenarios, field: str = "device_mb", question: str = "", reason: str = ""):
        super().__init__(name, scenarios, reason)
        self.field, self.question = field, question

    def evaluate(self, runs):
        peaks = [max((getattr(s, self.field) for s in r.steps if getattr(s, self.field) is not None), default=None) for r in runs]
        if None in peaks:
            return self.result(Status.SKIP, f"no {self.field} in one of the logs")
        return self.result(Status.LOOK, f"{self.field} peaks {', '.join(f'{n} {p} MB' for n, p in zip(self.scenarios, peaks))}. {self.question}")
