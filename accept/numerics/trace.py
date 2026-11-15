"""Where does a backward pass stop repeating? The gradient at every module's output is fingerprinted in two identical runs."""

from __future__ import annotations

import contextlib
from collections import Counter

from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .base import INFO, PASS, Finding
from .measure import digest
from .model import ModelCheck, deterministic_algorithms


class GradientTrace:
    """Records, in the order the backward reaches them, a digest of the gradient at each module's output."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.entries: list[tuple[str, tuple[int, int]]] = []
        self._handles: list = []

    def __enter__(self) -> GradientTrace:
        for name, child in self.module.named_modules():
            if name:
                self._handles.append(child.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *_) -> None:
        for handle in self._handles:
            handle.remove()

    def _hook(self, name: str):
        def hook(_module, _args, out) -> None:
            tensor = out if isinstance(out, Tensor) else next((t for t in out if isinstance(t, Tensor)), None) if isinstance(out, (tuple, list)) else None
            if tensor is not None and tensor.requires_grad:
                tensor.register_hook(lambda grad, n=name: self.entries.append((n, digest(grad))))
        return hook


class BackwardBisect(ModelCheck):
    name = "backward-bisect"
    purpose = "the first place, going backwards through the network, where two identical runs stop giving the same gradient"

    def run(self, ctx) -> list[Finding]:
        subject = ctx.subject
        inputs = subject.inputs()
        modules = dict(subject.denoiser.module().named_modules())
        found = []
        for label, scope, deterministic in (("default kernels", contextlib.nullcontext, False), ("math attention", lambda: sdpa_kernel(SDPBackend.MATH), False),
                                            ("math attention, deterministic algorithms", lambda: sdpa_kernel(SDPBackend.MATH), True)):
            traces = []
            with scope(), deterministic_algorithms(deterministic):
                for _ in range(2):
                    with GradientTrace(subject.denoiser.module()) as trace:
                        subject.evaluate(inputs)
                    traces.append(trace.entries)
            found.append(self._report(label, traces[0], traces[1], modules))
        return found

    def _report(self, label: str, a: list, b: list, modules: dict) -> Finding:
        if [n for n, _ in a] != [n for n, _ in b]:
            return Finding(self.name, label, INFO, f"the backward ran the modules in a different order ({len(a)} against {len(b)} gradients)")
        different = [i for i, ((_, x), (_, y)) in enumerate(zip(a, b, strict=True)) if x != y]
        if not different:
            return Finding(self.name, label, PASS, f"all {len(a)} module gradients are identical in both runs", {"differing": 0, "total": len(a)})
        first = different[0]
        before = a[first - 1][0] if first else "(the loss)"

        def kind(name: str) -> str:
            return type(modules[name]).__name__ if name in modules else "?"

        classes = Counter(kind(a[i][0]) for i in different).most_common(4)
        return Finding(self.name, label, INFO,
                       f"{len(different)} of {len(a)} module gradients differ; the first one going backwards is at the output of {a[first][0]} ({kind(a[first][0])}), "
                       f"whose predecessor {before} ({kind(before) if first else '-'}) was still identical: what sits between them is not repeatable; "
                       f"most affected: {', '.join(f'{c} x{n}' for c, n in classes)}",
                       {"differing": len(different), "total": len(a), "first": a[first][0], "before": before})
