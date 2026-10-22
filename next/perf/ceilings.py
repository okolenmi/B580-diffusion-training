"""What this card can do, measured now: the denominators every rating divides by."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass
class Ceilings:
    device: str = "cpu"
    torch_version: str = ""
    gemm_tflops: float = 0.0   # large bf16 matmul (fp32 on a CPU)
    int8_tops: float = 0.0     # torch._int_mm, 0 where unavailable
    copy_gbps: float = 0.0     # device-to-device copy, bytes read plus written
    dispatch_us: float = 0.0   # host time to issue one small op
    launch_us: float = 0.0     # time per small op when the queue is kept full (steady throughput)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Ceilings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CeilingProbe:
    """Measures the ceilings with short synchronised loops. ``quick`` uses small sizes (the CPU and the tests)."""

    def __init__(self, device: str = "cpu", quick: bool = False):
        self.device, self.quick = device, quick
        kind = device.split(":")[0]
        self.module = getattr(torch, kind, None) if kind != "cpu" else None

    def _sync(self) -> None:
        if self.module is not None and hasattr(self.module, "synchronize"):
            self.module.synchronize()

    def _time(self, fn, iters: int) -> float:
        for _ in range(max(2, iters // 4)):
            fn()
        self._sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        self._sync()
        return (time.perf_counter() - start) / iters

    def gemm(self) -> float:
        n = 256 if self.quick else 4096
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        a, b = torch.randn(n, n, device=self.device, dtype=dtype), torch.randn(n, n, device=self.device, dtype=dtype)
        return 2 * n ** 3 / self._time(lambda: a @ b, 3 if self.quick else 20) / 1e12

    def int8(self) -> float:
        if not hasattr(torch, "_int_mm") or self.device == "cpu":
            return 0.0
        n = 4096
        try:
            a = torch.randint(-127, 127, (n, n), device=self.device, dtype=torch.int8)
            b = torch.randint(-127, 127, (n, n), device=self.device, dtype=torch.int8)
            return 2 * n ** 3 / self._time(lambda: torch._int_mm(a, b), 20) / 1e12
        except Exception:  # noqa: BLE001 - a backend without the op simply has no int8 ceiling
            return 0.0

    def copy(self) -> float:
        n = (1 << 20) if self.quick else (1 << 27)
        src = torch.empty(n, device=self.device, dtype=torch.float16).normal_()
        dst = torch.empty_like(src)
        return 2 * src.numel() * src.element_size() / self._time(lambda: dst.copy_(src), 3 if self.quick else 20) / 1e9

    def launches(self) -> tuple[float, float]:
        x = torch.zeros(16, device=self.device)
        count = 200 if self.quick else 2000
        for _ in range(20):
            x.add_(1)
        self._sync()
        start = time.perf_counter()
        for _ in range(count):
            x.add_(1)
        issued = time.perf_counter() - start
        self._sync()
        total = time.perf_counter() - start
        return issued / count * 1e6, total / count * 1e6

    def run(self) -> Ceilings:
        dispatch, launch = self.launches()
        return Ceilings(self.device, torch.__version__, self.gemm(), self.int8(), self.copy(), dispatch, launch)


def load_or_measure(path: str | Path, device: str = "cpu", quick: bool = False) -> Ceilings:
    """The cached ceilings for this device and torch build, measured and stored when there are none."""
    path = Path(path)
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = f"{device}|{torch.__version__}|{'quick' if quick else 'full'}"
    if key in cache:
        return Ceilings.from_json(cache[key])
    measured = CeilingProbe(device, quick).run()
    cache[key] = measured.to_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1))
    return measured
