"""Micro-benchmarks: one operation at the shapes an SDXL step runs, timed on its own and rated against the measured ceilings.

Each ``Bench`` returns rows (name, shape, milliseconds, ideal, efficiency, note). They answer the questions a whole-step report
can only point at: is this GEMM shape slow, which attention backend is fastest and which can be captured, how much does a chain of
small kernels cost against one pass over the memory, what does a graph save per launch."""

from __future__ import annotations

import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch.nn import functional as F

from .ceilings import Ceilings, load_or_measure
from .clock import DeviceClock
from .cost import Cost
from .rating import Rating


@dataclass
class Row:
    group: str
    name: str
    shape: str
    ms: float | None
    ideal_ms: float | None = None
    efficiency: float | None = None
    bound: str | None = None
    note: str = ""


class BenchContext:
    """Sizes, device and ceilings shared by every bench. ``quick`` shrinks the sizes so the suite runs anywhere (a CPU, the tests)."""

    def __init__(self, device: str = "xpu", quick: bool = False, batch: int = 2, size: int = 768, ceilings: Ceilings | None = None,
                 backend: str = "auto"):
        self.device, self.quick = device, quick
        self.batch, self.size = (1, 64) if quick else (batch, size)
        self.iters = 2 if quick else 20
        self.dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.clock = DeviceClock(device)
        self.ceilings = ceilings or Ceilings(device=device)
        self.rating = Rating(self.ceilings)
        self.backend_choice = backend

    @property
    def latent(self) -> int:
        return max(self.size // 8, 2)

    def levels(self) -> list[tuple[str, int, int]]:
        """(name, channels, side) of the UNet's resolution levels."""
        side = self.latent
        if self.quick:
            return [("level 1", 32, side), ("level 2", 64, max(side // 2, 2)), ("level 3", 128, max(side // 4, 2))]
        return [("level 1", 320, side), ("level 2", 640, side // 2), ("level 3", 1280, side // 4)]

    def time_ms(self, fn: Callable[[], object]) -> float:
        for _ in range(max(2, self.iters // 4)):
            fn()
        self.clock.synchronize()
        start = time.perf_counter()
        for _ in range(self.iters):
            fn()
        self.clock.synchronize()
        return (time.perf_counter() - start) / self.iters * 1000

    def tensor(self, *shape, dtype=None, grad: bool = False) -> torch.Tensor:
        return torch.randn(*shape, device=self.device, dtype=dtype or self.dtype).requires_grad_(grad)

    def rated(self, group: str, name: str, shape: str, ms: float, cost: Cost, note: str = "") -> Row:
        ideal, bound = self.rating.ideal_ms(cost)
        return Row(group, name, shape, ms, ideal, ideal / ms if ms > 0 else None, bound, note)


class Bench:
    group = ""

    def rows(self, ctx: BenchContext) -> list[Row]:
        raise NotImplementedError


class GemmBench(Bench):
    """Matrix products of the sizes the attention projections and feed-forwards use."""

    group = "gemm"

    def rows(self, ctx):
        out = []
        for name, channels, side in ctx.levels():
            tokens = ctx.batch * side * side
            for label, k, n in (("projection", channels, channels), ("ff in", channels, channels * 8), ("ff out", channels * 4, channels), ("cross k/v", 2048 if not ctx.quick else 64, channels)):
                x, w = ctx.tensor(tokens, k), ctx.tensor(n, k)
                ms = ctx.time_ms(lambda: F.linear(x, w))
                e = x.element_size()
                out.append(ctx.rated(self.group, f"{name} {label}", f"{tokens}x{k} @ {k}x{n}", ms, Cost(2.0 * tokens * k * n, (tokens * k + tokens * n + n * k) * e)))
        return out


class ConvBench(Bench):
    """3x3 convolutions at each level: forward, and forward plus the input gradient (a frozen convolution has no weight gradient to compute)."""

    group = "convolution"

    def rows(self, ctx):
        out = []
        for name, channels, side in ctx.levels():
            conv = torch.nn.Conv2d(channels, channels, 3, padding=1).to(ctx.device, ctx.dtype).requires_grad_(False)
            x = ctx.tensor(ctx.batch, channels, side, side, grad=True)
            dy = torch.randn_like(x)
            flops = 2.0 * ctx.batch * channels * side * side * channels * 9
            e = x.element_size()
            data = (2 * x.numel() + conv.weight.numel()) * e
            shape = f"{ctx.batch}x{channels}x{side}x{side}"
            out.append(ctx.rated(self.group, f"{name} forward", shape, ctx.time_ms(lambda: conv(x)), Cost(flops, data)))
            out.append(ctx.rated(self.group, f"{name} forward+dx", shape, ctx.time_ms(lambda: conv(x).backward(dy)), Cost(2 * flops, 2 * data)))
        return out


class NormBench(Bench):
    """Group and layer normalisation: memory-bound, rated against the copy ceiling."""

    group = "norm"

    def rows(self, ctx):
        out = []
        for name, channels, side in ctx.levels():
            norm = torch.nn.GroupNorm(32 if channels % 32 == 0 else 1, channels).to(ctx.device, ctx.dtype)
            x = ctx.tensor(ctx.batch, channels, side, side, grad=True)
            dy = torch.randn_like(x)
            e, shape = x.element_size(), f"{ctx.batch}x{channels}x{side}x{side}"
            out.append(ctx.rated(self.group, f"{name} group norm forward", shape, ctx.time_ms(lambda: norm(x)), Cost(0.0, 2.0 * x.numel() * e)))
            out.append(ctx.rated(self.group, f"{name} group norm forward+dx", shape, ctx.time_ms(lambda: norm(x).backward(dy)), Cost(0.0, 5.0 * x.numel() * e)))
            tokens = ctx.batch * side * side
            layer = torch.nn.LayerNorm(channels).to(ctx.device, ctx.dtype)
            t = ctx.tensor(tokens, channels, grad=True)
            dt = torch.randn_like(t)
            out.append(ctx.rated(self.group, f"{name} layer norm forward+dx", f"{tokens}x{channels}", ctx.time_ms(lambda: layer(t).backward(dt)), Cost(0.0, 5.0 * t.numel() * e)))
        return out


class ElementwiseBench(Bench):
    """Single elementwise kernels against the copy ceiling, and a chain of them against one pass over the memory: the gap is what fusion would recover."""

    group = "elementwise"
    CHAIN = 8

    def rows(self, ctx):
        _, channels, side = ctx.levels()[1]
        n = ctx.batch * channels * side * side
        x, y, z = ctx.tensor(n), ctx.tensor(n), ctx.tensor(n)
        e, shape = x.element_size(), f"{n} elements"
        out = [ctx.rated(self.group, "silu", shape, ctx.time_ms(lambda: F.silu(x)), Cost(0.0, 2.0 * n * e)),
               ctx.rated(self.group, "add", shape, ctx.time_ms(lambda: x + y), Cost(0.0, 3.0 * n * e)),
               ctx.rated(self.group, "mul", shape, ctx.time_ms(lambda: x * y), Cost(0.0, 3.0 * n * e)),
               ctx.rated(self.group, "cast to float32", shape, ctx.time_ms(lambda: x.float()), Cost(0.0, n * (e + 4)))]

        def chain():
            h = x
            for _ in range(self.CHAIN):
                h = F.silu(h * y + z)
            return h
        out.append(ctx.rated(self.group, f"chain of {self.CHAIN} (mul, add, silu)", shape, ctx.time_ms(chain), Cost(0.0, 4.0 * n * e),
                             note="ideal is one fused pass (reads x, y, z, writes once)"))
        return out


class AttentionBench(Bench):
    """Scaled dot-product attention per backend: forward, forward+backward, and whether the forward+backward can be captured into a graph."""

    group = "attention"

    def rows(self, ctx):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        backends = [("default", None), ("math", SDPBackend.MATH), ("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION)]
        out = []
        dim = 64 if not ctx.quick else 16
        for name, channels, side in ctx.levels()[1:]:
            heads, tokens = channels // dim, side * side
            for kind, lk in (("self", tokens), ("cross", 77)):
                q, k, v = (ctx.tensor(ctx.batch, heads, n, dim, grad=True) for n in (tokens, lk, lk))
                dy = torch.randn(ctx.batch, heads, tokens, dim, device=ctx.device, dtype=ctx.dtype)
                flops = 4.0 * ctx.batch * heads * tokens * lk * dim
                shape = f"{ctx.batch}x{heads}x{tokens}x{dim} ({kind}, keys {lk})"
                for label, backend in backends:
                    try:
                        fwd = self._time(ctx, backend, lambda: F.scaled_dot_product_attention(q, k, v), sdpa_kernel)
                        both = self._time(ctx, backend, lambda: F.scaled_dot_product_attention(q, k, v).backward(dy), sdpa_kernel)
                    except Exception as error:  # noqa: BLE001 - a backend the device lacks is a result
                        out.append(Row(self.group, f"{name} {kind} {label}", shape, None, note=f"unavailable: {type(error).__name__}: {str(error).splitlines()[0][:100] if str(error) else ''}"))
                        continue
                    note = self._capture(ctx, backend, lambda: F.scaled_dot_product_attention(q, k, v).backward(dy), sdpa_kernel)
                    e = q.element_size()
                    data = (2 * q.numel() + 2 * k.numel()) * e
                    out.append(ctx.rated(self.group, f"{name} {kind} {label} forward", shape, fwd, Cost(flops, data)))
                    out.append(ctx.rated(self.group, f"{name} {kind} {label} forward+backward", shape, both, Cost(3.5 * flops, 3 * data), note=note))
        return out

    @staticmethod
    def _time(ctx, backend, fn, sdpa_kernel) -> float:
        if backend is None:
            return ctx.time_ms(fn)
        with sdpa_kernel(backend):
            return ctx.time_ms(fn)

    @staticmethod
    def _capture(ctx, backend, fn, sdpa_kernel) -> str:
        return GraphProbe(ctx).describe(fn, (lambda: sdpa_kernel(backend)) if backend is not None else None)


class GraphProbe:
    """Captures a callable into a device graph and times its replay; returns text, never raises."""

    def __init__(self, ctx: BenchContext):
        self.ctx = ctx
        self.module = getattr(torch, ctx.device.split(":")[0], None)

    @property
    def available(self) -> bool:
        return self.module is not None and hasattr(self.module, "XPUGraph" if self.ctx.device.startswith("xpu") else "CUDAGraph") and self.ctx.device != "cpu"

    def capture(self, fn: Callable[[], object], scope: Callable | None = None):
        graph_class = getattr(self.module, "XPUGraph", None) or getattr(self.module, "CUDAGraph")
        graph = graph_class()
        for _ in range(2):
            fn()
        self.ctx.clock.synchronize()
        with (scope() if scope else _nothing()), self.module.graph(graph):
            fn()
        self.ctx.clock.synchronize()
        return graph

    def replay_ms(self, graph) -> float:
        return self.ctx.time_ms(graph.replay)

    def describe(self, fn, scope=None) -> str:
        if not self.available:
            return "capture: no graph API on this device"
        try:
            graph = self.capture(fn, scope)
            return f"captures; replay {self.replay_ms(graph):.2f} ms"
        except Exception as error:  # noqa: BLE001
            return f"cannot be captured: {type(error).__name__}: {str(error).splitlines()[0][:100] if str(error) else ''}"


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class LayerBench(Bench):
    """A frozen linear layer, native against int8 (both gradient paths), at the shapes the UNet has: forward, and forward plus the input gradient."""

    group = "linear layer"
    LAYERS = (("attention projection", 1, 1), ("feed-forward in", 1, 8), ("feed-forward out", 4, 1), ("cross k/v", 0, 1))  # name, in and out features as multiples of the level width

    def rows(self, ctx):
        from ..families.general.quantize.backends import make_backend, shape_problem
        from ..families.general.quantize.ops import Int8LinearFn
        try:
            backend = make_backend(ctx.backend_choice, torch.device(ctx.device))
        except Exception as error:  # noqa: BLE001
            return [Row(self.group, "int8 backend", "", None, note=f"unavailable: {type(error).__name__}: {error}")]
        out = []
        for name, channels, side in ctx.levels()[1:]:
            tokens = ctx.batch * side * side
            for label, kmul, nmul in self.LAYERS:
                k = 2048 if label == "cross k/v" else channels * kmul
                n = channels * nmul
                if ctx.quick:
                    k, n = 256, 16 * nmul
                shape = f"{tokens}x{k} @ {k}x{n}"
                weight = (torch.randn(n, k, device=ctx.device) * 0.02).to(ctx.dtype)
                x = ctx.tensor(tokens, k, grad=True)
                dy = torch.randn(tokens, n, device=ctx.device, dtype=ctx.dtype)
                flops = 2.0 * tokens * k * n
                e = x.element_size()
                native = ctx.time_ms(lambda: F.linear(x, weight).backward(dy))
                out.append(ctx.rated(self.group, f"{name} {label} native forward+dx", shape, native, Cost(2 * flops, 2 * (tokens * (k + n) + n * k) * e)))
                problem = shape_problem(n, k)
                if problem:
                    out.append(Row(self.group, f"{name} {label} int8", shape, None, note=f"stays native: {problem}"))
                    continue
                w8, scale = backend.prepare(weight)
                for grad in ("dense", "int8"):
                    ms = ctx.time_ms(lambda g=grad: Int8LinearFn.apply(x, w8, scale, None, backend, g).backward(dy))
                    out.append(ctx.rated(self.group, f"{name} {label} int8 ({grad} dx) forward+dx", shape, ms,
                                         Cost(2 * flops, 2 * tokens * (k + n) * e + 2 * n * k, low_precision=True), note=f"{native / ms:.2f}x of native speed"))
                out.append(ctx.rated(self.group, f"{name} {label} weight dequantize", f"{n}x{k}", ctx.time_ms(lambda: backend.dequant(w8, scale)), Cost(0.0, n * k * (1 + e))))
        return out


class LaunchBench(Bench):
    """A chain of tiny kernels run eagerly and replayed from a graph: the launch cost a graph removes, per kernel."""

    group = "launch"
    CHAIN = 200

    def rows(self, ctx):
        x = ctx.tensor(1024)

        def chain():
            h = x
            for _ in range(self.CHAIN):
                h = h + 1
            return h
        eager = ctx.time_ms(chain)
        out = [Row(self.group, f"{self.CHAIN} tiny kernels, eager", "1024 elements", eager, note=f"{eager * 1000 / self.CHAIN:.1f} us per kernel")]
        probe = GraphProbe(ctx)
        if not probe.available:
            return out + [Row(self.group, "graph replay", "", None, note="no graph API on this device")]
        try:
            replay = probe.replay_ms(probe.capture(chain))
            out.append(Row(self.group, f"{self.CHAIN} tiny kernels, graph replay", "1024 elements", replay, note=f"{replay * 1000 / self.CHAIN:.1f} us per kernel, {eager / replay:.1f}x faster than eager"))
        except Exception as error:  # noqa: BLE001
            out.append(Row(self.group, "graph replay", "", None, note=f"cannot capture: {type(error).__name__}: {str(error).splitlines()[0][:100] if str(error) else ''}"))
        return out


SUITES: dict[str, type[Bench]] = {"gemm": GemmBench, "conv": ConvBench, "norm": NormBench, "elementwise": ElementwiseBench,
                                  "attention": AttentionBench, "layers": LayerBench, "launch": LaunchBench}


@dataclass
class BenchReport:
    meta: dict
    rows: list[Row] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"meta": self.meta, "rows": [vars(r) for r in self.rows], "errors": self.errors}

    def markdown(self) -> str:
        m = self.meta
        c = m["ceilings"]
        out = [f"# Micro-benchmarks ({m['device']}, batch {m['batch']}, {m['size']} px)", "",
               f"Ceilings: gemm {c['gemm_tflops']:.1f} TFLOPS, int8 {c['int8_tops']:.1f} TOPS, copy {c['copy_gbps']:.0f} GB/s. "
               "Efficiency is ideal time over measured time, ideal from the ceiling of whatever bounds the operation.", ""]
        for group in dict.fromkeys(r.group for r in self.rows):
            out += [f"## {group}", "", "| operation | shape | ms | ideal ms | efficiency | bound | note |", "|---|---|---:|---:|---:|---|---|"]
            for r in sorted((r for r in self.rows if r.group == group), key=lambda r: (r.efficiency is None, r.efficiency or 0.0)):
                f = lambda v, s=".3f": "-" if v is None else format(v, s)  # noqa: E731
                out.append(f"| {r.name} | {r.shape} | {f(r.ms)} | {f(r.ideal_ms)} | {'-' if r.efficiency is None else format(r.efficiency, '.0%')} | {r.bound or '-'} | {r.note} |")
            out.append("")
        if self.errors:
            out += ["## Suites that failed", ""] + [f"- {k}: {v}" for k, v in self.errors.items()]
        return "\n".join(out) + "\n"


class BenchRunner:
    """Runs the chosen suites, one failure never stopping the rest, and writes bench.md and bench.json."""

    def __init__(self, ctx: BenchContext, names: list[str] | None = None, say: Callable[[str], None] = print):
        unknown = [n for n in names or [] if n not in SUITES]
        if unknown:
            raise ValueError(f"unknown benchmark suite {unknown[0]!r}; choose from {', '.join(SUITES)}")
        self.ctx, self.names, self.say = ctx, names or list(SUITES), say

    def run(self) -> BenchReport:
        ctx = self.ctx
        report = BenchReport({"device": ctx.device, "batch": ctx.batch, "size": ctx.size, "quick": ctx.quick, "torch": torch.__version__, "ceilings": ctx.ceilings.to_json()})
        for name in self.names:
            self.say(f"bench: {name}")
            try:
                report.rows += SUITES[name]().rows(ctx)
            except Exception as error:  # noqa: BLE001 - recorded, the other suites still run
                report.errors[name] = f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}"
        return report

    def write(self, report: BenchReport, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "bench.json").write_text(json.dumps(report.to_json(), indent=1, default=str))
        (folder / "bench.md").write_text(report.markdown())
        return folder / "bench.md"


def context_for(device: str, quick: bool, folder: str | Path, batch: int = 2, size: int = 768, backend: str = "auto") -> BenchContext:
    ceilings = load_or_measure(Path(folder) / "ceilings.json", device, quick=quick)
    return BenchContext(device, quick, batch, size, ceilings, backend)
