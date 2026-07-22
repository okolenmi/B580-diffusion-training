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
step() call -- AND, since the buffer now lives in a
nodes.memory.manager.MemoryManager instead of a plain torch.empty() call
made fresh each time, it's reused *across* step() calls too (allocated
once, then only regrown if a later step's largest parameter is bigger
than what's already held).

Routing the buffer through MemoryManager instead of managing it as a raw
attribute here is a deliberate choice, not just "use the new utility
class because it exists": the reset-vs-free asymmetry bug found in
core/optimizers.py's legacy classes (`_scratch`/`_pool` cleared in
free_states() but not reset_states() in some classes -- see
docs/nodes_package_design.md's "Course correction" section) is exactly
the class of bug a hand-rolled `self._scratch` attribute here would be
one careless edit away from reintroducing. Centralizing acquire/release/
free through one reviewed class means there's exactly one place that
logic can go wrong, not N ad hoc copies of it -- see
nodes/memory/manager.py's module docstring for the full reasoning.

What this strategy does NOT do yet, deliberately not silently skipped:
  - No torch.xpu.MemPool integration (the legacy ChunkedXPUAdafactor/
    ChunkedXPUCAME wrap their scratch buffer's allocation in one, to
    reduce allocator fragmentation) -- MemoryManager.get_buffer() is
    exactly the seam that integration would go through later (wrap the
    torch.empty() call inside it in a MemPool context), without this
    strategy needing to change at all -- real, valuable, separate
    follow-up.
  - Passes its buffer to Algorithm.compute_update()'s `scratch` parameter,
    but no Algorithm implemented so far (CAMEAlgorithm included) actually
    uses it for its own internal intermediates yet -- so the *sole*
    memory saving from this strategy right now is the gradient-cast
    reuse described above (now also cached across steps), not the deeper
    savings the legacy verified fix achieved. Real, but partial, stated
    precisely rather than implied to be the full picture.
"""

from __future__ import annotations

import torch

from ...memory.manager import MemoryManager
from .base import ExecutionStrategy

_SCRATCH_TAG = "grad_cast"


class ChunkedScratchBufferStrategy(ExecutionStrategy):

    def __init__(self, memory: MemoryManager | None = None):
        """memory: inject a shared MemoryManager (e.g. if a future caller
        wants several strategies/handles to share one memory budget). A
        strategy owns its own private instance by default -- matches
        ComposedOptimizerHandle's existing pattern of each handle owning
        its own strategy instance, so there's no implicit global state
        either way.
        """
        self.memory = memory if memory is not None else MemoryManager()

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
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
                update = algorithm.compute_update(grad_view, states[i], scratch=grad_view)
                p.data.add_(update.to(dtype=p.dtype), alpha=-param_lr[i])
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
