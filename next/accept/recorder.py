"""Records what selected modules output and what gradient reaches their outputs, for a later comparison with another run of the same network."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

from torch import Tensor, nn

from ..perf.modules import selected_kind


@dataclass
class Recording:
    outputs: dict[str, Tensor] = field(default_factory=dict)
    grads: dict[str, Tensor] = field(default_factory=dict)
    dropped: int = 0  # modules left out because the memory budget was spent

    @property
    def megabytes(self) -> float:
        return sum(t.numel() * t.element_size() for d in (self.outputs, self.grads) for t in d.values()) / 2**20


def is_block(module: nn.Module) -> bool:
    return selected_kind(module) == "block"


class BlockRecorder:
    """Hooks the modules ``select`` accepts (blocks by default). Tensors are kept on the host in fp32 so a comparison does not depend on the
    device. Modules called more than once keep the first call. ``limit_mb`` caps what is kept; the rest is counted in ``dropped``."""

    def __init__(self, model: nn.Module, select: Callable[[nn.Module], bool] = is_block, limit_mb: float = 2048.0):
        self.model, self.select, self.limit_mb = model, select, limit_mb
        self.recording = Recording()
        self._handles: list = []

    def __enter__(self) -> BlockRecorder:
        for path, module in self.model.named_modules():
            if path and self.select(module):
                self._handles.append(module.register_forward_hook(self._hook(path)))
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _room(self, tensor: Tensor) -> bool:
        return self.recording.megabytes + tensor.numel() * 4 / 2**20 <= self.limit_mb

    def _hook(self, path: str):
        def hook(module, args, output):
            if not isinstance(output, Tensor) or path in self.recording.outputs:
                return
            if not self._room(output):
                self.recording.dropped += 1
                return
            self.recording.outputs[path] = output.detach().float().cpu()
            if output.requires_grad:
                output.register_hook(lambda grad, p=path: self._keep_grad(p, grad))
        return hook

    def _keep_grad(self, path: str, grad: Tensor) -> None:
        if path not in self.recording.grads and self._room(grad):
            self.recording.grads[path] = grad.detach().float().cpu()
