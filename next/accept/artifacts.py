"""What a finished run left behind, read back as data: the console log as step records and named lines, the perf report, the output files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

STEP = re.compile(r"^step (\d+)/(\d+)\s+loss (\S+)\s+(\d+) ms")
SHAPE = re.compile(r"\[(\d+)x(\d+)x(\d+)( new)?\]")
GRAPHS = re.compile(r"graphs (\d+) \((\d+) replays\)")
MEMORY = re.compile(r"device (\d+) MB \(live (\d+), step peak (\d+), cached (\d+)\)")
GRAPH_LINE = re.compile(r"^\[graph\] shape (\(.*?\)): replay (\d+) ms against (\d+) ms eager(; the graph does not pay and is dropped)?")
QUANT = re.compile(r"^(\w+) base weights \((\w+)\): (\d+) layers, (-?\d+) MB saved")
BUDGET = re.compile(r"^memory budget: (\d+) MB of the card's (\d+) MB")


@dataclass
class StepLine:
    step: int
    total: int
    loss: float
    ms: int
    shape: tuple[int, int, int] | None = None
    new_shape: bool = False
    graphs: int | None = None
    replays: int | None = None
    device_mb: int | None = None
    live_mb: int | None = None
    peak_mb: int | None = None
    cached_mb: int | None = None


@dataclass
class GraphDecision:
    shape: str
    replay_ms: int
    eager_ms: int
    dropped: bool


@dataclass
class RunArtifacts:
    folder: Path
    lines: list[str] = field(default_factory=list)
    steps: list[StepLine] = field(default_factory=list)
    graph_decisions: list[GraphDecision] = field(default_factory=list)
    quantization: dict | None = None
    memory_limit_mb: int | None = None
    finished: bool = False
    failed_with: str | None = None
    report: dict | None = None

    @classmethod
    def read(cls, folder: str | Path) -> RunArtifacts:
        folder = Path(folder)
        art = cls(folder)
        log = folder / "run.log"
        art.lines = log.read_text(encoding="utf8", errors="replace").splitlines() if log.exists() else []
        for line in art.lines:
            art._parse(line)
        report = folder / "report.json"
        art.report = json.loads(report.read_text()) if report.exists() else None
        return art

    def _parse(self, line: str) -> None:
        if (m := STEP.match(line)):
            record = StepLine(int(m[1]), int(m[2]), float(m[3]), int(m[4]))
            if (s := SHAPE.search(line)):
                record.shape, record.new_shape = (int(s[1]), int(s[2]), int(s[3])), bool(s[4])
            if (g := GRAPHS.search(line)):
                record.graphs, record.replays = int(g[1]), int(g[2])
            if (d := MEMORY.search(line)):
                record.device_mb, record.live_mb, record.peak_mb, record.cached_mb = (int(d[i]) for i in range(1, 5))
            self.steps.append(record)
        elif (m := GRAPH_LINE.match(line)):
            self.graph_decisions.append(GraphDecision(m[1], int(m[2]), int(m[3]), bool(m[4])))
        elif (m := QUANT.match(line)):
            self.quantization = {"label": m[1], "backend": m[2], "layers": int(m[3]), "saved_mb": int(m[4])}
        elif (m := BUDGET.match(line)) and self.memory_limit_mb is None:
            self.memory_limit_mb = int(m[1])
        elif line.startswith("run: ok"):
            self.finished = True
        elif line.startswith("run: failed"):
            self.failed_with = line

    def tail(self, count: int = 6) -> str:
        return "\n".join(self.lines[-count:]) if self.lines else "(no log)"

    def settled_steps(self, skip: int = 0) -> list[StepLine]:
        """Steps that did not pay for a first-seen shape, after the first ``skip`` steps."""
        return [s for s in self.steps[skip:] if not s.new_shape]

    def files(self, pattern: str) -> list[Path]:
        return sorted(self.folder.glob(pattern))
