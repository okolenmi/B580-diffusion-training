"""ForeachApplyStrategy: batches an ExecutionStrategy's *apply* step via
torch._foreach_* ops, grouped by (device, dtype).

Per-parameter algorithm math still runs one Algorithm.compute_update()
call at a time -- see algorithms/base.py, that part isn't shape-uniform
across an arbitrary parameter set and batching it would be an
algorithm-specific concern (the legacy ForeachXPUAdafactor's own
"factored" path already falls back to a plain per-parameter loop for
exactly this reason -- confirmed by reading it directly). What this
strategy batches is the part every Algorithm shares regardless of its own
math: applying `decay` then `delta` to `param.data`, done as at most two
foreach_* calls per (device, dtype) group instead of two Python-level
tensor ops per parameter -- a real reduction in kernel-launch/interpreter
overhead for a graph with many small parameters (LoRA's normal case).
The batched-apply logic itself now lives in base.py's
apply_updates_batched() -- extracted there once
ShapeGroupedForeachStrategy (shape_grouped_foreach.py) needed the
identical logic too, rather than a second copy of it.

No scratch-buffer/MemoryManager reuse yet, unlike ChunkedScratchBufferStrategy
-- this is the first slice, proving the batched-apply idea composes
correctly; buffer reuse on top of it is real, separate follow-up work.

Verified bit-exact (torch.equal(), not a tolerance) vs. SimpleLoopStrategy
on CPU across CAME/Adafactor/AdamW, float32 and bf16 -- see
nodes/smoke_tests/smoke_test_foreach_strategy_equivalence.py. Getting
there required a real fix, not just running the test until it passed:
`torch._foreach_mul_`'s ScalarList overload (a bare `list[float]`) takes a
different internal rounding path than eager `.mul_()` for bf16 tensors --
diverged in ~94% of direct random trials -- while passing each scalar as
a 0-dim float32 tensor instead matched exactly. `torch._foreach_sub_` had
no such issue (0 mismatches, same trial count). See
base.py's apply_updates_batched() docstring for the fix itself.
"""

from __future__ import annotations

from .base import ExecutionStrategy, apply_updates_batched


class ForeachApplyStrategy(ExecutionStrategy):

    def step(self, algorithm, params, states, param_lr, n_steps: int = 1) -> None:
        algorithm.begin_step(n_steps)
        entries = []
        for i, p in enumerate(params):
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            delta, decay = algorithm.compute_update(grad, p, states[i], param_lr[i])
            entries.append((p, delta, decay))
        apply_updates_batched(entries)

    def zero_grad(self, params) -> None:
        for p in params:
            if p.grad is not None:
                p.grad = None
