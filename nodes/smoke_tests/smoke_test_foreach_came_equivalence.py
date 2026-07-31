"""Numerical equivalence: core.optimizers.ForeachXPUCAME vs ChunkedXPUCAME.

Same formulas (see ForeachXPUCAME's own docstring for exactly what
changed and why: scalars stay 0-dim tensors instead of being converted
with float(), and the clip_div != 1.0 Python-level branch became an
unconditional tensor divide). Not expected to be torch.equal() bit-exact
-- a tensor divide and a Python-float divide can round differently in the
last bit even for the same mathematical operation -- so this checks
torch.allclose() with a tight tolerance across many steps, on both
factored (2D) and non-factored (1D) parameters, weight decay on and off,
and confirms parameters actually moved (not just "stayed close to init
and both matched trivially").
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from core.optimizers import ChunkedXPUCAME, ForeachXPUCAME

DEVICE = "cpu"


def run(weight_decay: float, n_steps: int = 30):
    torch.manual_seed(3)
    shapes = [(37, 53), (17, 4), (129,)]  # factored + 1D, deliberately non-round
    p_init = [torch.randn(s, device=DEVICE) for s in shapes]

    p_chunked = [p.clone().requires_grad_(False) for p in p_init]
    p_foreach = [p.clone().requires_grad_(False) for p in p_init]

    opt_chunked = ChunkedXPUCAME(p_chunked, lr=1e-3, weight_decay=weight_decay, device=DEVICE)
    opt_foreach = ForeachXPUCAME(p_foreach, lr=1e-3, weight_decay=weight_decay, device=DEVICE)

    gen = torch.Generator().manual_seed(11)
    for step in range(n_steps):
        grads = [torch.randn(s, generator=gen) for s in shapes]
        for p, g in zip(p_chunked, grads):
            p.grad = g.clone()
        for p, g in zip(p_foreach, grads):
            p.grad = g.clone()

        opt_chunked.step()
        opt_foreach.step()

        for i, (pc, pf) in enumerate(zip(p_chunked, p_foreach)):
            assert torch.allclose(pc, pf, atol=1e-5, rtol=1e-4), (
                f"step {step}, param {i} (shape {shapes[i]}): diverged. "
                f"max abs diff = {(pc - pf).abs().max().item()}"
            )

    for p_orig, p_final in zip(p_init, p_chunked):
        assert not torch.allclose(p_orig, p_final), "params never moved -- test isn't exercising anything"

    return p_chunked, p_foreach


def main():
    print("[ForeachXPUCAME vs ChunkedXPUCAME: weight_decay=0.0]")
    run(weight_decay=0.0)
    print("    PASS: 30 steps, 3 params (2D + 1D shapes), allclose throughout")

    print("[ForeachXPUCAME vs ChunkedXPUCAME: weight_decay=0.01]")
    run(weight_decay=0.01)
    print("    PASS")

    print()
    print("=" * 60)
    print("SMOKE TEST: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
