"""Can two processes use the same weights that sit once on the card? Gates for the shared-weights design in rework/76-parallel-runs.md.
Each gate runs in its own process(es) so a crash or a hang of the driver costs one row, not the run; every row says what happened in words."""

from __future__ import annotations

import ctypes.util
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

MB = 1 << 20
PATTERN_MB = 256


@dataclass
class Row:
    gate: str
    status: str  # ok | failed | skipped
    detail: str = ""
    numbers: dict = field(default_factory=dict)


def _torch(device: str):
    import torch
    return torch, getattr(torch, device)


def _pattern(torch, device: str, mb: int):
    n = mb * MB // 4
    return (torch.arange(n, device=device, dtype=torch.float32) % 251).contiguous()


def _free_mb(torch, mod) -> float | None:
    info = getattr(mod, "mem_get_info", None)
    return info()[0] / MB if info else None


def _owner(device: str, send, ready, release, done) -> None:
    """Owner process of gates 1-3: allocates, hands the tensor over, frees it when told to."""
    torch, mod = _torch(device)
    try:
        t = _pattern(torch, device, PATTERN_MB)
        mod.synchronize()
        before = _free_mb(torch, mod)
        send.put(t)
        ready.set()
        release.wait(120)
        del t
        getattr(mod, "empty_cache", lambda: None)()
        mod.synchronize()
        done.put({"free_mb_with_tensor": before, "free_mb_after_release": _free_mb(torch, mod)})
    except Exception:
        done.put({"error": traceback.format_exc(limit=3)})
        ready.set()


def _borrower(device: str, recv, got, release, report) -> None:
    """Borrower of gates 1-3: reads the tensor, does arithmetic on it, tells what it still sees after the owner let go."""
    torch, mod = _torch(device)
    out: dict = {}
    try:
        t = recv.get(timeout=60)
        out["shape"] = list(t.shape)
        out["device"] = str(t.device)
        out["checksum_ok"] = bool(abs(float(t.sum()) - float(_pattern(torch, device, PATTERN_MB).sum())) < 1.0)
        x = torch.randn(1024, 1024, device=device)
        out["matmul_on_shared"] = bool(torch.isfinite((t[: 1024 * 1024].view(1024, 1024) @ x).sum()))
        got.set()
        release.set()
        time.sleep(3)
        try:
            out["after_owner_released_ok"] = bool(abs(float(t.sum()) - float(_pattern(torch, device, PATTERN_MB).sum())) < 1.0)
        except Exception as exc:
            out["after_owner_released_ok"] = False
            out["after_owner_released_error"] = repr(exc)[:300]
    except Exception:
        out["error"] = traceback.format_exc(limit=3)
        got.set()
        release.set()
    report.put(out)


def _loop(device: str, seconds: float, report, start) -> None:
    torch, mod = _torch(device)
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    b = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    mod.synchronize()
    start.wait(60)
    n = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        for _ in range(20):
            a @ b
        mod.synchronize()
        n += 20
    report.put(n / seconds)


class IpcProbe:
    """The gates, in the order the design needs them answered."""

    def __init__(self, device: str = "xpu", quick: bool = False) -> None:
        self.device = device
        self.seconds = 3.0 if quick else 10.0
        self.ctx = mp.get_context("spawn")

    def run(self) -> list[Row]:
        rows = [self.inventory()]
        sharing = self.sharing()
        rows += sharing
        rows += [self.two_processes(), self.round_trip()]
        return rows

    def inventory(self) -> Row:
        try:
            torch, mod = _torch(self.device)
            from torch.multiprocessing import reductions
            t = torch.zeros(4, device=self.device)
            info = {
                "torch": torch.__version__,
                "device_count": mod.device_count(),
                "dlpack_device": list(t.__dlpack_device__()),
                "reductions_mention_device": [n for n in dir(reductions) if self.device in n.lower()],
                "level_zero_loader": ctypes.util.find_library("ze_loader"),
                "ipc_event_classes": [c.__module__ + "." + c.__name__ for c in getattr(reductions, "ipc_event_classes", [])],
            }
            return Row("inventory", "ok", "what this build offers for sharing", info)
        except Exception:
            return Row("inventory", "failed", traceback.format_exc(limit=3))

    def sharing(self) -> list[Row]:
        """Gates 1-3 in one pair of processes: hand-over, use, and the owner letting go while the borrower still holds it."""
        q, report, done = self.ctx.Queue(), self.ctx.Queue(), self.ctx.Queue()
        ready, got, release = self.ctx.Event(), self.ctx.Event(), self.ctx.Event()
        owner = self.ctx.Process(target=_owner, args=(self.device, q, ready, release, done))
        borrower = self.ctx.Process(target=_borrower, args=(self.device, q, got, release, report))
        owner.start()
        borrower.start()
        try:
            seen = report.get(timeout=180)
            freed = done.get(timeout=60)
        except Exception as exc:
            seen, freed = {"error": f"no answer: {exc!r}"}, {}
        finally:
            for p in (owner, borrower):
                p.join(10)
                if p.is_alive():
                    p.kill()
        if "error" in seen or "error" in freed:
            why = freed.get("error") or seen.get("error")
            return [Row("G1 hand a device tensor to another process", "failed", why),
                    Row("G2 use it read-only", "skipped"), Row("G3 owner frees while borrowed", "skipped")]
        return [
            Row("G1 hand a device tensor to another process", "ok", f"arrived on {seen['device']}, shape {seen['shape']}", {"checksum_ok": seen["checksum_ok"]}),
            Row("G2 use it read-only", "ok" if seen["matmul_on_shared"] and seen["checksum_ok"] else "failed", "arithmetic on the borrowed tensor"),
            Row("G3 owner frees while borrowed", "ok" if seen["after_owner_released_ok"] else "failed",
                "borrower still reads valid data" if seen["after_owner_released_ok"] else seen.get("after_owner_released_error", "borrower saw wrong data"), freed),
        ]

    def two_processes(self) -> Row:
        """Gate 4: what time-slicing between two processes costs (needs no sharing)."""
        def run(count: int) -> list[float]:
            report, start = self.ctx.Queue(), self.ctx.Event()
            procs = [self.ctx.Process(target=_loop, args=(self.device, self.seconds, report, start)) for _ in range(count)]
            for p in procs:
                p.start()
            time.sleep(8)
            start.set()
            rates = [report.get(timeout=self.seconds + 120) for _ in procs]
            for p in procs:
                p.join(10)
            return rates
        try:
            alone = run(1)[0]
            pair = run(2)
            return Row("G4 two processes on one card", "ok", "bf16 2048 matmul, iterations per second",
                       {"alone": alone, "together_each": pair, "together_sum_vs_alone": sum(pair) / alone})
        except Exception:
            return Row("G4 two processes on one card", "failed", traceback.format_exc(limit=3))

    def round_trip(self) -> Row:
        """Gate 5: what a process that needs parked weights waits for (device -> host -> device, 1 GB, pinned and not)."""
        try:
            torch, mod = _torch(self.device)
            t = _pattern(torch, self.device, 1024)
            numbers = {}
            for name, pinned in (("pinned", True), ("pageable", False)):
                host = torch.empty(t.shape, dtype=t.dtype, pin_memory=pinned and self.device != "cpu")
                mod.synchronize()
                t0 = time.perf_counter()
                host.copy_(t)
                mod.synchronize()
                down = time.perf_counter() - t0
                t0 = time.perf_counter()
                back = host.to(self.device)
                mod.synchronize()
                up = time.perf_counter() - t0
                numbers[name] = {"down_s": down, "up_s": up, "up_GB_s": 1.0 / up}
                del back
            return Row("G5 park and reload 1 GB", "ok", "the wait of a process whose weights were parked", numbers)
        except Exception:
            return Row("G5 park and reload 1 GB", "failed", traceback.format_exc(limit=3))

    @staticmethod
    def markdown(rows: list[Row]) -> str:
        out = ["# Shared weights across processes: gates", ""]
        for r in rows:
            out.append(f"* **{r.gate}**: {r.status}. {r.detail}")
            if r.numbers:
                out.append("  `" + json.dumps(r.numbers) + "`")
        return "\n".join(out) + "\n"

    def write(self, rows: list[Row], folder: str) -> Path:
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        (out / "ipc.json").write_text(json.dumps([asdict(r) for r in rows], indent=1))
        (out / "ipc.md").write_text(self.markdown(rows))
        return out / "ipc.md"
