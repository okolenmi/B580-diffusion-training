"""Builds the report (a plain dict that is also the JSON form) and writes it as report.md, report.json and optionally trace.json."""

from __future__ import annotations

import json
from pathlib import Path

from .budget import StepRecord, build_budget
from .ceilings import Ceilings
from .findings import findings_for
from .modules import ModuleRecord
from .rating import BACKWARD_FACTOR, Rating
from .trace import TraceResult

TOP_KERNELS = 25
TOP_MODULES = 30


class ReportBuilder:
    def __init__(self, ceilings: Ceilings, device: str, meta: dict | None = None):
        self.ceilings, self.device, self.meta = ceilings, device, meta or {}
        self.rating = Rating(ceilings)

    def build(self, steps: list[StepRecord], trace: TraceResult, records: list[ModuleRecord], clean: list[StepRecord] | None = None,
              seconds: dict[str, float] | None = None) -> dict:
        eager = [s for s in steps if not s.replayed]
        budget = build_budget(steps, trace, clean)
        count = max(len(steps), 1)
        total = trace.device_us or 1.0
        categories = [{"category": name, "calls": c["calls"] / count, "device_ms": c["device_us"] / 1000 / count, "share": c["device_us"] / total}
                      for name, c in sorted(trace.by_category().items(), key=lambda kv: -kv[1]["device_us"])]
        kernels = [{"name": r.name, "category": r.category, "calls": r.calls / count, "device_ms": r.device_us / 1000 / count,
                    "us_per_call": r.device_us / r.calls if r.calls else 0.0} for r in trace.rows[:TOP_KERNELS]]
        # kernels outside the top list still count in the categories; the top list is for reading
        modules = self._modules(records, len(eager))
        groups = self._groups(modules)
        report = {"meta": {"device": self.device, "ceilings": self.ceilings.to_json(), "steps_in_window": len(steps),
                           "eager_steps": len(eager), "replayed_steps": len(steps) - len(eager),
                           "device_time_is_host_time": trace.device_is_host, "report_seconds": {k: round(v, 2) for k, v in (seconds or {}).items()}, **self.meta},
                  "budget": budget, "categories": categories, "kernels": kernels, "modules": modules, "groups": groups,
                  "reconciliation": self._reconcile(budget, modules, len(steps) - len(eager))}
        report["findings"] = [vars(f) for f in findings_for(report)]
        return report

    def _modules(self, records: list[ModuleRecord], eager_steps: int) -> list[dict]:
        n = max(eager_steps, 1)
        rows = []
        for r in records:
            own = (r.self_forward_ms + r.self_recompute_ms + r.self_backward_ms) / n
            row = {"path": r.path, "kind": r.kind, "region": r.region, "role": r.role, "int8": r.int8, "calls": r.calls / n,
                   "fwd_ms": r.forward_ms / n, "recompute_ms": r.recompute_ms / n, "bwd_ms": r.backward_ms / n,
                   "self_ms": own, "ideal_ms": None, "recoverable_ms": None, "efficiency": None, "bound": None,
                   "flops": None, "bytes": None, "shapes": [list(s) for s in r.shapes]}
            if r.cost is not None and own > 0:
                ideal, bound = self.rating.ideal_ms(r.cost)
                factor = BACKWARD_FACTOR.get(r.kind, 1.0)
                ideal_total = ideal * (r.calls + r.recompute_calls + factor * r.calls) / n
                rated = self.rating.rate(own, ideal_total, bound)
                row.update(ideal_ms=ideal_total, recoverable_ms=rated.recoverable_ms, efficiency=rated.efficiency, bound=bound,
                           flops=r.cost.flops, bytes=r.cost.bytes)
            rows.append(row)
        return sorted(rows, key=lambda m: -m["self_ms"])

    def _groups(self, modules: list[dict]) -> list[dict]:
        merged: dict[tuple, dict] = {}
        for m in modules:
            if m["ideal_ms"] is None:
                continue
            g = merged.setdefault((m["kind"], m["role"], m["region"], m["int8"]),
                                  {"kind": m["kind"], "role": m["role"], "region": m["region"], "int8": m["int8"], "count": 0,
                                   "total_ms": 0.0, "ideal_ms": 0.0, "recoverable_ms": 0.0, "memory_ideal_ms": 0.0, "compute_ideal_ms": 0.0})
            g["count"] += 1
            g["total_ms"] += m["self_ms"]
            g["ideal_ms"] += m["ideal_ms"]
            g["recoverable_ms"] += m["recoverable_ms"]
            g["compute_ideal_ms" if m["bound"] == "compute" else "memory_ideal_ms"] += m["ideal_ms"]
        for g in merged.values():
            g["efficiency"] = g["ideal_ms"] / g["total_ms"] if g["total_ms"] else 0.0
            g["bound"] = "compute" if g["compute_ideal_ms"] >= g["memory_ideal_ms"] else "memory"
        return sorted(merged.values(), key=lambda g: -g["recoverable_ms"])

    @staticmethod
    def _reconcile(budget: dict, modules: list[dict], replayed: int) -> dict:
        if replayed or not modules:
            return {"module_ms": None, "device_ms": budget["device_ms"], "unrated_ms": None}
        own = sum(m["self_ms"] for m in modules)
        unrated = sum(m["self_ms"] for m in modules if m["ideal_ms"] is None)
        return {"module_ms": own, "device_ms": budget["device_ms"], "unrated_ms": unrated, "wall_ms": budget["wall_ms"]}


def _fmt(value, spec: str = ".1f", empty: str = "-") -> str:
    return empty if value is None else format(value, spec)


class ReportWriter:
    def __init__(self, folder: str | Path):
        self.folder = Path(folder)

    def write(self, report: dict) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / "report.json").write_text(json.dumps(report, indent=1, default=str))
        (self.folder / "report.md").write_text(self.markdown(report))
        return self.folder / "report.md"

    def markdown(self, report: dict) -> str:
        b, meta = report["budget"], report["meta"]
        out = [f"# Performance report ({meta['device']})", ""]
        if meta["device_time_is_host_time"]:
            out += ["> No device activity was recorded (a CPU run): kernel times below are host operator times and utilisation is not meaningful.", ""]
        out += [f"Window: {meta['steps_in_window']} steps ({meta['eager_steps']} eager, {meta['replayed_steps']} replayed). All times are per step.", "",
                "## Step budget", "",
                f"- step {b['wall_ms']:.0f} ms, median of {b['timed_steps']} " + ("unprofiled steps" if b["wall_clean"] else "profiled steps (inflated by the profiler)")
                + f" (best {b['wall_ms_min']:.0f}); eager {_fmt(b['eager_ms'], '.0f')} ms, replayed {_fmt(b['replayed_ms'], '.0f')} ms",
                f"- the profiled window's steps took {_fmt(b['profiled_ms'], '.0f')} ms each (hooks and profiler); device and module times come from them",
                f"- device busy {b['device_ms']:.0f} ms" + (f", utilisation {b['utilisation']:.0%}" if b["utilisation"] is not None else ""),
                f"- {b['launches']:.0f} kernel launches, {b['ops']:.0f} operators, {b['host_us_per_launch']:.0f} us host time per launch"]
        if b.get("peak_mb") is not None:
            out.append(f"- step peak memory {b['peak_mb']:.0f} MB")
        if b["phase_ms"]:
            out.append("- phases: " + ", ".join(f"{k} {v:.0f} ms" for k, v in b["phase_ms"].items()))
        out += ["", "## Findings, largest first", ""]
        out += [f"{i}. **{f['title']}** ({f['severity_ms']:.0f} ms)\n   - {f['evidence']}\n   - lever: {f['lever']}" for i, f in enumerate(report["findings"], 1)] \
            or ["Nothing crossed a rule's threshold."]
        out += ["", "## Device time by category", "", "| category | calls | ms | share |", "|---|---:|---:|---:|"]
        out += [f"| {c['category']} | {c['calls']:.0f} | {c['device_ms']:.1f} | {c['share']:.0%} |" for c in report["categories"]]
        c = meta["ceilings"]
        out += ["", "## Measured ceilings (what 'ideal' means here)", "",
                f"gemm {_fmt(c.get('gemm_tflops'))} TFLOPS, int8 {_fmt(c.get('int8_tops'))} TOPS, copy {_fmt(c.get('copy_gbps'))} GB/s, "
                f"dispatch {_fmt(c.get('dispatch_us'))} us, queued launch {_fmt(c.get('launch_us'))} us. Backward costs are estimates (see perf/rating.py).",
                "", "## Slowest piece groups by recoverable time", "",
                "| kind | role | region | layers | ms | ideal ms | efficiency | bound | recoverable ms |", "|---|---|---|---:|---:|---:|---:|---|---:|"]
        out += [f"| {g['kind']}{' int8' if g['int8'] else ''} | {g['role']} | {g['region']} | {g['count']} | {g['total_ms']:.1f} | {g['ideal_ms']:.1f} | "
                f"{g['efficiency']:.0%} | {g['bound']} | {g['recoverable_ms']:.1f} |" for g in report["groups"][:TOP_MODULES]]
        out += ["", "## Slowest modules (own time, nested modules excluded)", "",
                "| module | kind | calls | fwd | recompute | bwd | own ms | ideal | efficiency |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        out += [f"| {m['path']} | {m['kind']} | {m['calls']:.0f} | {m['fwd_ms']:.1f} | {m['recompute_ms']:.1f} | {m['bwd_ms']:.1f} | {m['self_ms']:.1f} | "
                f"{_fmt(m['ideal_ms'])} | {_fmt(None if m['efficiency'] is None else m['efficiency'] * 100, '.0f')}{'' if m['efficiency'] is None else '%'} |"
                for m in report["modules"][:TOP_MODULES]]
        r = report["reconciliation"]
        out += ["", "## Reconciliation", "",
                f"device {r['device_ms']:.0f} ms; modules {_fmt(r['module_ms'], '.0f')} ms (unrated {_fmt(r['unrated_ms'], '.0f')} ms). "
                "Module times need eager steps: a window with replayed steps has no module table." if r["module_ms"] is None else
                f"device {r['device_ms']:.0f} ms; modules' own time {r['module_ms']:.0f} ms, of which {r['unrated_ms']:.0f} ms is in modules with no cost model; wall {r['wall_ms']:.0f} ms.",
                "", "## Slowest kernels", "", "| kernel | category | calls | ms | us per call |", "|---|---|---:|---:|---:|"]
        out += [f"| `{k['name'][:90]}` | {k['category']} | {k['calls']:.0f} | {k['device_ms']:.2f} | {k['us_per_call']:.0f} |" for k in report["kernels"]]
        return "\n".join(out) + "\n"
