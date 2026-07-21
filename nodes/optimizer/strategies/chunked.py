"""ChunkedScratchBufferStrategy: shares one scratch buffer across parameters.

Implements one specific, honest slice of memory optimization -- see
docs/nodes_package_design.md's follow-up notes for the two-axis
distinction this is built around: reusing *one* buffer across *different
parameters* (this strategy) vs. an Algorithm reusing a buffer for its *own
internal* sequential intermediates (a separate, Algorithm-specific
concern -- see algorithms/base.py's compute_update() docstring and
algorithms/came.py's current honest non-implementation of it).

What this strategy actually saves: without it, `p.grad.detach().float()`
allocates a fresh tensor for every parameter, every step. With it, one
buffer sized to the largest managed parameter is reused (via `.copy_()`,
an in-place write) for every parameter's gradient in turn, within one
step() call.

What this strategy does NOT do yet, deliberately not silently skipped:
  - No torch.xpu.MemPool integration (the legacy ChunkedXPUAdafactor/
    ChunkedXPUCAME wrap their scratch buffer's allocation in one, to
    reduce allocator fragmentation) -- real, valuable, separate follow-up.
  - The buffer is allocated fresh on each step() call, not cached across
    calls the way the legacy classes' lazy _init_scratch does -- the
    memory-saving property (one buffer sized to max param, not N buffers)
    holds either way, but caching would avoid repeated allocator overhead
    across steps -- also real, separate follow-up.
  - Passes its buffer to Algorithm.compute_update()'s `scratch` parameter,
    but no Algorithm implemented so far (CAMEAlgorithm included) actually
    uses it for its own internal intermediates yet -- so the *sole*
    memory saving from this strategy right now is the gradient-cast
    reuse described above, not the deeper savings the legacy verified fix
    achieved. Real, but partial -- stated precisely rather than implied
    to be the full picture.
"""

from __future__ import annotations

import torch

from .base import ExecutionStrategy


class ChunkedScratchBufferStrategy(ExecutionStrategy):

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        grad_params = [(i, p) for i, p in enumerate(params) if p.grad is not None]
        if not grad_params:
            return
        max_numel = max(p.numel() for _, p in grad_params)
        device = grad_params[0][1].device
        scratch = torch.empty(max_numel, dtype=torch.float32, device=device)

        for i, p in grad_params:
            n = p.numel()
            grad_view = scratch[:n].reshape(p.shape)
            grad_view.copy_(p.grad.detach())
            update = algorithm.compute_update(grad_view, states[i], scratch=grad_view)
            p.data.add_(update.to(dtype=p.dtype), alpha=-param_lr[i])

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
