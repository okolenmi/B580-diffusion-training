"""A few known operations under the profiler, with the raw events and the attribution printed: tells whether this build's profiler
links kernel launches to operators (correlation ids) before a long run depends on it."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.autograd.profiler import record_function

from .events import ANNOTATION
from .trace import KernelTrace

FIELDS = ("name", "device_type", "activity_type", "start_ns", "duration_ns", "start_thread_id", "correlation_id", "linked_correlation_id",
          "external_id", "shapes", "dtypes")
SAMPLE = 60


class EventsProbe:
    def __init__(self, device: str = "xpu"):
        self.device = device

    def _work(self) -> None:
        dev = torch.device(self.device)
        x = torch.randn(2048, 1024, device=dev, dtype=torch.bfloat16)
        w = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
        f = torch.randn(2048, 1024, device=dev)
        for _ in range(3):
            with record_function(f"{ANNOTATION}probe.linear|forward"):
                y = x @ w
            with record_function(f"{ANNOTATION}probe.cast|forward"):
                z = f.to(torch.bfloat16)
            with record_function(f"{ANNOTATION}probe.add|forward"):
                z = (y + z).float()
        getattr(torch, self.device.split(":")[0], torch.cuda).synchronize() if self.device != "cpu" else None

    def run(self) -> dict:
        self._work()  # kernels get built outside the profile
        trace = KernelTrace(self.device, provenance=True)
        with trace:
            self._work()
        kineto = trace._profiler.profiler.kineto_results
        sample = []
        for event in kineto.events():
            row = {}
            for field in FIELDS:
                try:
                    value = getattr(event, field)()
                    row[field] = value if isinstance(value, (int, float, str, list)) else str(value)
                except Exception as error:  # noqa: BLE001 - a field this build lacks is a finding, not a failure
                    row[field] = f"<{type(error).__name__}>"
            sample.append(row)
        sample.sort(key=lambda r: (r.get("device_type") == "DeviceType.CPU", r.get("start_ns") or 0))
        prov = trace.result.provenance
        return {"device": self.device, "kinds": prov.kinds, "coverage": prov.coverage, "rows": prov.rows[:20],
                "modules": {k: v for k, v in prov.modules.items()}, "sample": sample[:SAMPLE]}

    def write(self, data: dict, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "events.json").write_text(json.dumps(data, indent=1, default=str))
        cov = data["coverage"]
        lines = [f"# Profiler event probe ({data['device']})", "",
                 f"Kernel time tied to an operator: {cov['operator']:.0%}; to a module range: {cov['module']:.0%} (of {cov['kernel_ms']:.2f} ms).", "",
                 "Event kinds: " + ", ".join(f"{k} x{v}" for k, v in sorted(data["kinds"].items())), "",
                 "| kernel | operator | dtypes | shapes | module | calls | us |", "|---|---|---|---|---|---:|---:|"]
        lines += [f"| `{r['kernel'][:50]}` | {r['op']} | {r['dtypes']} | {r['shapes']} | {r['module']} | {r['calls']} | {r['device_us']:.0f} |" for r in data["rows"]]
        path = folder / "events.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf8")
        return path
