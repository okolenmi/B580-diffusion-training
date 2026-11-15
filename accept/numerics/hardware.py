"""The device and the driver, without any model: do copies and repeated launches give the same bits?"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from .base import Finding, NumericCheck
from .measure import differing

MB = 2**20


def _sync(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    if hasattr(backend, "synchronize"):
        backend.synchronize()


class TransferRoundtrip(NumericCheck):
    name = "transfers"
    purpose = "host to device and back, pageable and pinned, comes back bit for bit (offload and reload depend on it)"

    SIZES_MB = (1, 64, 256)
    REPEATS = 8

    def run(self, ctx) -> list[Finding]:
        if ctx.device.type == "cpu":
            return [self.skip("copies", "the device is the CPU")]
        gen = torch.Generator().manual_seed(7)
        found = []
        for dtype in (torch.bfloat16, torch.float32):
            for size in self.SIZES_MB:
                n = size * MB // torch.empty((), dtype=dtype).element_size()
                source = torch.randn(n, generator=gen).to(dtype)
                for pinned in (False, True):
                    try:
                        host = source.pin_memory() if pinned else source
                    except Exception as error:  # noqa: BLE001 - pinning is not available everywhere
                        found.append(self.skip(f"{dtype} {size} MB pinned", f"cannot pin: {type(error).__name__}"))
                        continue
                    bad, seconds = 0, 0.0
                    for _ in range(self.REPEATS):
                        start = time.perf_counter()
                        back = host.to(ctx.device).cpu()
                        _sync(ctx.device)
                        seconds += time.perf_counter() - start
                        bad += differing(back, source)
                    gbps = 2 * self.REPEATS * size / 1024 / seconds
                    f = self.exact(f"{str(dtype).split('.')[-1]} {size} MB {'pinned' if pinned else 'pageable'}", bad, n * self.REPEATS)
                    f.detail += f" ({gbps:.1f} GB/s both ways)"
                    found.append(f)
        return found


class DeviceMemoryPattern(NumericCheck):
    name = "memory"
    purpose = "a large block of device memory keeps what was written to it (random data, then its complement), across other work"

    BLOCK_MB = 256

    def run(self, ctx) -> list[Finding]:
        if ctx.device.type == "cpu":
            return [self.skip("pattern", "the device is the CPU")]
        total = ctx.memory_mb
        count = max(total // self.BLOCK_MB, 1)
        gen = torch.Generator().manual_seed(11)
        blocks, host = [], []
        for _ in range(count):
            pattern = torch.randint(-2**31, 2**31 - 1, (self.BLOCK_MB * MB // 4,), generator=gen, dtype=torch.int32)
            host.append(pattern)
            blocks.append(pattern.to(ctx.device))
        _sync(ctx.device)
        busy = torch.randn(2048, 2048, device=ctx.device, dtype=torch.bfloat16)
        for _ in range(50):
            busy = (busy @ busy).tanh()  # other work on the card between the write and the read
        _sync(ctx.device)
        bad = sum(differing(b.cpu(), h) for b, h in zip(blocks, host, strict=True))
        first = self.exact(f"{count * self.BLOCK_MB} MB random pattern", bad, count * self.BLOCK_MB * MB // 4, "words")
        flipped = 0
        for b, h in zip(blocks, host, strict=True):
            b.bitwise_not_()
            flipped += differing(b.cpu(), ~h)
        second = self.exact("the same memory, complemented in place", flipped, count * self.BLOCK_MB * MB // 4, "words")
        return [first, second]


class OpRepeatability(NumericCheck):
    name = "repeat"
    purpose = "the same operation on the same inputs, launched again and again, gives the same bits"

    REPEATS = 25

    def run(self, ctx) -> list[Finding]:
        device, dtype = ctx.device, torch.bfloat16
        gen = torch.Generator().manual_seed(3)

        def rnd(*shape, grad=False, dt=dtype):
            return (torch.randn(*shape, generator=gen) * 0.5).to(device, dt).requires_grad_(grad)

        a, b = rnd(4096, 1280), rnd(1280, 5120)
        a32, b32 = rnd(2048, 1280, dt=torch.float32), rnd(1280, 1280, dt=torch.float32)
        bx, by = rnd(20, 1024, 64), rnd(20, 64, 1024)
        x, w = rnd(2, 320, 64, 64, grad=True), rnd(320, 320, 3, 3, grad=True)
        gx, gw, gb = rnd(2, 640, 32, 32, grad=True), rnd(640), rnd(640)
        ln_x, ln_w = rnd(4096, 1280, grad=True), rnd(1280)
        q, k, v = rnd(2, 10, 1024, 64, grad=True), rnd(2, 10, 1024, 64, grad=True), rnd(2, 10, 1024, 64, grad=True)
        sm = rnd(2, 10, 1024, 1024)
        dy = {}

        def with_grad(fn, *leaves):
            def run():
                for t in leaves:
                    t.grad = None
                out = fn()
                g = dy.setdefault(out.shape, rnd(*out.shape))
                out.backward(g)
                return [out.detach(), *[t.grad.detach() for t in leaves]]
            return run

        from torch.nn.attention import SDPBackend, sdpa_kernel

        def math_attention():
            with sdpa_kernel(SDPBackend.MATH):
                return F.scaled_dot_product_attention(q, k, v)

        forward = {"gemm bf16 4096x1280x5120": lambda: [a @ b], "gemm fp32 2048x1280x1280": lambda: [a32 @ b32],
                   "batched gemm bf16": lambda: [bx @ by], "softmax bf16": lambda: [sm.softmax(-1)]}
        backward = {"conv3x3 forward+backward": with_grad(lambda: F.conv2d(x, w, padding=1), x, w),
                    "group norm forward+backward": with_grad(lambda: F.group_norm(gx, 32, gw, gb), gx),
                    "layer norm forward+backward": with_grad(lambda: F.layer_norm(ln_x, (1280,), ln_w), ln_x),
                    "attention (default kernel) forward+backward": with_grad(lambda: F.scaled_dot_product_attention(q, k, v), q, k, v),
                    "attention (math backend) forward+backward": with_grad(math_attention, q, k, v)}
        found = []
        for name, fn in {**forward, **backward}.items():
            first = [t.detach().clone() for t in fn()]
            bad = 0
            for _ in range(self.REPEATS):
                bad += sum(differing(got, ref) > 0 for got, ref in zip(fn(), first, strict=True))
            total = self.REPEATS * len(first)
            if name in forward:
                found.append(self.exact(name, bad, total, "results"))
            else:  # a nondeterministic backward is known behaviour of some kernels: shown, and a WARN
                f = self.measured(name, f"{bad} of {total} results differ from the first launch", {"differing": bad, "total": total}, warn_if=bad > 0)
                found.append(f)
        return found + self._long_reductions(ctx)

    LONG = ((1024, 10240, 1280), (4096, 10240, 1280), (1024, 1280, 10240), (4096, 1280, 5120))  # rows x reduction x columns: gemms of a feed-forward block, forward and backward to its input

    def _long_reductions(self, ctx) -> list[Finding]:
        """Gemms with a long reduction axis: a library may split it across work groups and add the parts atomically, which is not repeatable."""
        from .model import deterministic_algorithms

        gen = torch.Generator().manual_seed(4)
        found = []
        for m, k, n in self.LONG:
            a = (torch.randn(m, k, generator=gen) * 0.5).to(ctx.device, torch.bfloat16)
            b = (torch.randn(k, n, generator=gen) * 0.5).to(ctx.device, torch.bfloat16)
            for deterministic in (False, True):
                with deterministic_algorithms(deterministic):
                    first = (a @ b).clone()
                    bad = sum(differing(a @ b, first) > 0 for _ in range(self.REPEATS))
                suffix = ", deterministic algorithms" if deterministic else ""
                found.append(self.measured(f"gemm bf16 {m}x{k}x{n}{suffix}", f"{bad} of {self.REPEATS} results differ from the first launch",
                                           {"differing": bad, "total": self.REPEATS}, warn_if=bad > 0 and deterministic))
        return found


class GraphReplay(NumericCheck):
    name = "graph"
    purpose = "a captured forward and backward replays to the bits the eager run gives (the trainer's graph mode rests on this)"

    def run(self, ctx) -> list[Finding]:
        device = ctx.device
        backend = getattr(torch, device.type, None)
        if not hasattr(backend, "XPUGraph"):
            return [self.skip("replay", "this device cannot capture graphs")]
        torch.manual_seed(5)
        net = torch.nn.Sequential(torch.nn.Linear(1024, 2048), torch.nn.GELU(), torch.nn.Linear(2048, 1024)).to(device, torch.bfloat16)
        params = list(net.parameters())
        for p in params:
            p.grad = torch.zeros_like(p)
        static = torch.zeros(256, 1024, device=device, dtype=torch.bfloat16)
        dy = torch.randn(256, 1024, device=device).to(torch.bfloat16)
        inputs = [torch.randn(256, 1024, device=device).to(torch.bfloat16) for _ in range(3)]

        def step():
            net(static).backward(dy)

        def eager(value):
            for p in params:
                p.grad.zero_()
            static.copy_(value)
            step()
            return [p.grad.clone() for p in params]

        for value in inputs:  # warm the kernels before capture
            eager(value)
        _sync(device)
        graph = backend.XPUGraph()
        with backend.graph(graph, pool=backend.graph_pool_handle()):
            step()
        _sync(device)
        found = []
        for index, value in enumerate(inputs):
            want = eager(value)
            for p in params:
                p.grad.zero_()
            static.copy_(value)
            graph.replay()
            _sync(device)
            bad = sum(differing(p.grad, ref) for p, ref in zip(params, want, strict=True))
            found.append(self.exact(f"replay {index + 1} against eager", bad, sum(p.numel() for p in params), "gradient values"))
        return found
