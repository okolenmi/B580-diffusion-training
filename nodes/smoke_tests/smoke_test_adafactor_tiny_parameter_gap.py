"""Investigates exactly how much of a gap AdafactorAlgorithm has for
parameters under 10,000 elements against each of the three legacy
Adafactor classes -- written because reading the source revealed the
three don't actually agree with each other on tiny-parameter handling,
which changes what "close the gap" even means. Not a pass/fail smoke
test in the usual sense -- prints findings for each class separately so
a human can read the actual numbers, run this directly and send the
output back rather than trusting a summary.

Three genuinely different mechanisms found by reading core/optimizers.py
directly (not assumed from any docstring or comment):

  - ForeachXPUAdafactor (core/optimizers.py ~923-1143): NO tiny-parameter
    special case at all -- `_step_factored`/`_step_unfactored` use the
    same row/col factored / 1D EMA formula AdafactorAlgorithm already
    implements, for every parameter regardless of size. Hypothesis this
    script tests: ForeachAdafactorOptimizerNode might already be fully
    equivalent to ComposedAdafactorOptimizerNode(strategy="foreach"),
    with no algorithm change needed at all.

  - FusedXPUAdafactor (core/optimizers.py ~1149-1389, TINY_NUMEL=10_000):
    a real per-parameter special case -- plain elementwise second-moment
    EMA (self._tiny_vs_map[i]) instead of the row/col factored
    approximation, computed independently per parameter (each backward
    hook only ever sees its own parameter's gradient). Self-contained --
    could become an AdafactorAlgorithm.compute_update() branch on its
    own, IF nothing else needed to match Foreach's behavior of not
    special-casing at all (see "why this can't just be added
    universally" below).

  - ChunkedXPUAdafactor (core/optimizers.py ~101-397, hardcoded 10_000):
    a DIFFERENT mechanism -- every tiny parameter across the WHOLE
    optimizer gets flattened and concatenated into one shared tensor,
    with ONE shared clip (rms_g computed over the concatenation, not per
    parameter) and ONE shared EMA state (self._tiny_vs, sized to the
    total concatenated numel). This is a cross-parameter batching
    concern, not a per-parameter algorithm formula -- AdafactorAlgorithm
    operates one parameter at a time by design (see algorithms/base.py)
    and has no way to see other parameters in the same optimizer.
    Replicating this exactly would need new ExecutionStrategy-level
    machinery (something like ShapeGroupedBatchStrategy, but grouping by
    "under the size threshold" instead of "same shape" -- a real,
    separate feature, not a formula fix). Not attempted here.

Why this script doesn't just add the FusedXPUAdafactor-style branch to
AdafactorAlgorithm directly: that Algorithm is shared across every
strategy (chunked/foreach/simple/shape_grouped/fused). Adding a
universal per-parameter tiny branch would fix ComposedFusedAdafactorOptimizerNode's
match against FusedXPUAdafactor -- but would then make
ComposedAdafactorOptimizerNode(strategy="foreach") start DIVERGING from
ForeachXPUAdafactor, which currently matches specifically because
neither side special-cases tiny parameters. The three legacy references
disagree with each other, so one shared Algorithm literally cannot
match all three for tiny parameters at once without becoming
strategy-aware about it, which is a real design decision, not
implemented here -- this script's job is only to pin down the actual
numbers so that decision can be made with real information instead of
a guess.

Run this directly: `python nodes/smoke_tests/smoke_test_adafactor_tiny_parameter_gap.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.optimizers import ChunkedXPUAdafactor, ForeachXPUAdafactor, FusedXPUAdafactor
from nodes.optimizer.algorithms.adafactor import AdafactorAlgorithm
from nodes.optimizer.composed import ComposedOptimizerHandle
from nodes.optimizer.composed_fused import ComposedFusedOptimizerHandle
from nodes.optimizer.strategies.chunked import ChunkedScratchBufferStrategy
from nodes.optimizer.strategies.foreach import ForeachApplyStrategy

DEVICE = "cpu"  # every legacy class falls back to plain tensors off-XPU
                # (torch.xpu.MemPool() construction is wrapped in
                # try/except Exception -- confirmed by reading
                # ChunkedXPUAdafactor._init_scratch() directly), so this
                # runs the same math without needing XPU hardware present.


def _small_params(seed=0):
    """A factored (2D) and an unfactored (1D) parameter, both well under
    the 10,000-element TINY_NUMEL threshold every legacy class that has
    one uses -- 20*30=600 and 64 elements respectively."""
    g = torch.Generator().manual_seed(seed)
    p_factored = torch.randn(20, 30, generator=g)
    p_unfactored = torch.randn(64, generator=g)
    return p_factored, p_unfactored


def _max_abs_diff(a, b):
    return (a - b).abs().max().item()


def check_foreach_hypothesis(dtype, momentum, n_steps=5, seed=0):
    """Hypothesis: ForeachXPUAdafactor has no tiny-parameter special
    case, so it should already match AdafactorAlgorithm+ForeachApplyStrategy
    for small parameters -- no gap, no algorithm change needed."""
    torch.manual_seed(seed)
    p0_f, p0_u = _small_params(seed)
    beta1 = 0.9 if momentum else None

    legacy_f = torch.nn.Parameter(p0_f.clone().to(dtype))
    legacy_u = torch.nn.Parameter(p0_u.clone().to(dtype))
    legacy = ForeachXPUAdafactor([legacy_f, legacy_u], lr=1e-3, beta1=beta1,
                                  weight_decay=0.0, scale_parameter=False, device=DEVICE)

    new_f = torch.nn.Parameter(p0_f.clone().to(dtype))
    new_u = torch.nn.Parameter(p0_u.clone().to(dtype))
    algorithm = AdafactorAlgorithm(beta1=beta1, weight_decay=0.0, scale_parameter=False)
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=ForeachApplyStrategy(),
                                      params=[new_f, new_u], lr=1e-3, device=DEVICE)

    torch.manual_seed(seed + 1)
    for step in range(n_steps):
        g_f = torch.randn_like(p0_f).to(dtype)
        g_u = torch.randn_like(p0_u).to(dtype)
        legacy_f.grad = g_f.clone()
        legacy_u.grad = g_u.clone()
        legacy.step()
        new_f.grad = g_f.clone()
        new_u.grad = g_u.clone()
        handle.step()

    diff_f = _max_abs_diff(legacy_f.data, new_f.data)
    diff_u = _max_abs_diff(legacy_u.data, new_u.data)
    print(f"    dtype={dtype} momentum={momentum}: "
          f"factored max_abs_diff={diff_f:.3e}, unfactored max_abs_diff={diff_u:.3e}")
    return diff_f, diff_u


def check_fused_tiny_gap(dtype, momentum, n_steps=5, seed=0):
    """Characterizes the real gap: FusedXPUAdafactor's tiny-parameter
    elementwise EMA vs. AdafactorAlgorithm's regular factored/1D math
    (no tiny branch), for the same small parameters.

    Drives both through real backward() calls (register_post_accumulate_grad_hook
    only fires as part of autograd's actual gradient-accumulation step, not
    on a manually-assigned .grad) -- the gradient itself is injected as an
    exact value via `(p * g_target).sum()`, since d/dp of that is exactly
    g_target, so the "random gradient" drawn each step is still exactly
    what both sides see, just produced through a real backward pass."""
    torch.manual_seed(seed)
    p0_f, p0_u = _small_params(seed)
    beta1 = 0.9 if momentum else None

    legacy_f = torch.nn.Parameter(p0_f.clone().to(dtype))
    legacy_u = torch.nn.Parameter(p0_u.clone().to(dtype))
    legacy = FusedXPUAdafactor([legacy_f, legacy_u], lr=1e-3, beta1=beta1,
                                weight_decay=0.0, scale_parameter=False, device=DEVICE)
    legacy.register_hooks()

    new_f = torch.nn.Parameter(p0_f.clone().to(dtype))
    new_u = torch.nn.Parameter(p0_u.clone().to(dtype))
    algorithm = AdafactorAlgorithm(beta1=beta1, weight_decay=0.0, scale_parameter=False)
    handle = ComposedFusedOptimizerHandle(algorithm=algorithm, params=[new_f, new_u],
                                           lr=1e-3, device=DEVICE)

    torch.manual_seed(seed + 1)
    for step in range(n_steps):
        g_f = torch.randn_like(p0_f).to(dtype)
        g_u = torch.randn_like(p0_u).to(dtype)

        legacy.begin_step(1)  # legacy FusedXPUAdafactor has its own begin_step()
        # too (core/optimizers.py ~1302) -- without calling it each iteration,
        # _in_backward never resets after step 1 and self.t/rho_t freeze.
        (legacy_f * g_f).sum().add((legacy_u * g_u).sum()).backward()
        handle.begin_step(1)  # required before each logical step's backward() --
        # not automatic, see composed_fused.py's own module docstring
        (new_f * g_f).sum().add((new_u * g_u).sum()).backward()

    diff_f = _max_abs_diff(legacy_f.data, new_f.data)
    diff_u = _max_abs_diff(legacy_u.data, new_u.data)
    print(f"    dtype={dtype} momentum={momentum}: "
          f"factored max_abs_diff={diff_f:.3e}, unfactored max_abs_diff={diff_u:.3e}")
    return diff_f, diff_u


def check_chunked_tiny_gap(dtype, n_steps=5, seed=0):
    """Characterizes the cross-parameter batching gap: ChunkedXPUAdafactor
    ties multiple tiny parameters together into one shared clip/EMA state
    -- confirm AdafactorAlgorithm (genuinely per-parameter, no way to see
    other parameters) diverges, and roughly by how much, rather than just
    asserting it must."""
    torch.manual_seed(seed)
    p0_f, p0_u = _small_params(seed)

    legacy_f = torch.nn.Parameter(p0_f.clone().to(dtype))
    legacy_u = torch.nn.Parameter(p0_u.clone().to(dtype))
    legacy = ChunkedXPUAdafactor([legacy_f, legacy_u], lr=1e-3, beta1=None,
                                  weight_decay=0.0, scale_parameter=False, device=DEVICE)

    new_f = torch.nn.Parameter(p0_f.clone().to(dtype))
    new_u = torch.nn.Parameter(p0_u.clone().to(dtype))
    algorithm = AdafactorAlgorithm(beta1=None, weight_decay=0.0, scale_parameter=False)
    handle = ComposedOptimizerHandle(algorithm=algorithm, strategy=ChunkedScratchBufferStrategy(),
                                      params=[new_f, new_u], lr=1e-3, device=DEVICE)

    torch.manual_seed(seed + 1)
    for step in range(n_steps):
        g_f = torch.randn_like(p0_f).to(dtype)
        g_u = torch.randn_like(p0_u).to(dtype)
        legacy_f.grad = g_f.clone()
        legacy_u.grad = g_u.clone()
        legacy.step()
        new_f.grad = g_f.clone()
        new_u.grad = g_u.clone()
        handle.step()

    diff_f = _max_abs_diff(legacy_f.data, new_f.data)
    diff_u = _max_abs_diff(legacy_u.data, new_u.data)
    print(f"    dtype={dtype}: factored max_abs_diff={diff_f:.3e}, "
          f"unfactored max_abs_diff={diff_u:.3e} (expected to diverge -- "
          f"this is the cross-parameter batching gap, documented not fixed)")
    return diff_f, diff_u


def main():
    print("=== A: ForeachXPUAdafactor hypothesis (expected: near-zero diff, no gap) ===")
    for dtype in (torch.float32, torch.bfloat16):
        for momentum in (False, True):
            check_foreach_hypothesis(dtype, momentum)

    print("\n=== B: FusedXPUAdafactor tiny-path gap (expected: real, nonzero diff) ===")
    for dtype in (torch.float32, torch.bfloat16):
        for momentum in (False, True):
            check_fused_tiny_gap(dtype, momentum)

    print("\n=== C: ChunkedXPUAdafactor tiny-path gap, for reference "
          "(expected: real, nonzero diff -- cross-parameter batching, not attempted) ===")
    for dtype in (torch.float32, torch.bfloat16):
        check_chunked_tiny_gap(dtype)

    print("\nDone. Please send back this full output -- the actual numbers "
          "decide what happens next, not just pass/fail.")


if __name__ == "__main__":
    main()
