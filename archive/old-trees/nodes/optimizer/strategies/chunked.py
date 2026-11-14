"""ChunkedScratchBufferStrategy: shares one scratch buffer across parameters.

One specific, honest slice of memory optimization: reusing *one* buffer
across *different parameters* (this strategy) vs. an Algorithm reusing a
buffer for its *own internal* sequential intermediates (a separate,
Algorithm-specific concern -- see algorithms/base.py's compute_update()
docstring).

What this strategy actually saves: without it, `p.grad.detach().float()`
allocates a fresh tensor for every parameter, every step. With it, one
buffer sized to the largest managed parameter is reused (via `.copy_()`,
an in-place write) for every parameter's gradient in turn, within one
step() call -- AND, since the buffer lives in a
nodes.memory.manager.MemoryManager instead of a plain torch.empty() call
made fresh each time, it's reused *across* step() calls too (allocated
once, then only regrown if a later step's largest parameter is bigger
than what's already held).

Routing the buffer through MemoryManager instead of managing it as a raw
attribute here avoids the reset-vs-free asymmetry bug class MemoryManager
itself exists to prevent -- see nodes/memory/manager.py's module
docstring.

What this strategy passes its buffer to Algorithm.compute_update()'s
`scratch` parameter, but whether a given Algorithm actually uses that for
its own internal intermediates is up to the Algorithm -- see each
Algorithm's own module docstring for whether it does.
"""

from __future__ import annotations

import torch

from ...memory.manager import MemoryManager
from .base import ExecutionStrategy

_SCRATCH_TAG = "grad_cast"


class ChunkedScratchBufferStrategy(ExecutionStrategy):

    def __init__(self, memory: MemoryManager | None = None, use_mempool: bool = False):
        """memory: inject a shared MemoryManager (e.g. if a future caller
        wants several strategies/handles to share one memory budget). A
        strategy owns its own private instance by default -- matches
        ComposedOptimizerHandle's existing pattern of each handle owning
        its own strategy instance, so there's no implicit global state
        either way.

        use_mempool: passed straight through to the default MemoryManager
        (ignored if an explicit `memory` is given -- set it on that
        instance instead). Default off -- see nodes/memory/manager.py's
        module docstring for real, documented tradeoffs before enabling.
        """
        self.memory = memory if memory is not None else MemoryManager(use_mempool=use_mempool)

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        algorithm.begin_step(n_steps)
        grad_params = [(i, p) for i, p in enumerate(params) if p.grad is not None]
        if not grad_params:
            return
        max_numel = max(p.numel() for _, p in grad_params)
        device = grad_params[0][1].device
        scratch = self.memory.get_buffer(_SCRATCH_TAG, max_numel, torch.float32, device)
        try:
            for i, p in grad_params:
                n = p.numel()
                grad_view = scratch[:n].reshape(p.shape)
                grad_view.copy_(p.grad.detach())
                delta, decay = algorithm.compute_update(grad_view, p, states[i], param_lr[i],
                                                          scratch=grad_view)
                if decay is not None:
                    p.data.mul_(decay)
                p.data.sub_(delta.to(dtype=p.dtype))
        finally:
            self.memory.release(_SCRATCH_TAG)

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None

    def offload_extra(self) -> None:
        """The scratch buffer is now cached across step() calls (that's
        the whole point of routing it through MemoryManager instead of
        allocating fresh every time) -- so unlike the old fresh-alloc-
        every-call version, it genuinely holds device memory between
        steps now, and MUST be freed here or offloading this handle to
        free VRAM would silently miss this part of it. Exactly the
        reset-vs-free asymmetry bug class this module's docstring
        references -- freeing through MemoryManager.free_all() means
        there's one place this gets handled, not a line that's easy to
        forget in this method specifically."""
        self.memory.free_all()

    def reload_extra(self, device) -> None:
        """Nothing to restore -- get_buffer() lazily reallocates on
        whatever device the next step() call actually runs on."""
        pass

    def free_extra(self) -> None:
        self.memory.free_all()
