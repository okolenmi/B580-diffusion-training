#!/usr/bin/env python3
"""How much does computing the LoRA branch in bf16 instead of fp32 change the
output? NO GPU, NO MODEL -- one real LoRALinear, CPU, both dtypes.

The launch count says the fused form is cheaper (see
measure_lora_launches.py: -7.2% unhoisted, -22.8% with the A/B casts cached
per step). It does not say whether the two compute the same thing, and they do
not exactly: today's forward casts the *activation* up to fp32 for the rank-64
matmul and casts the result back down; the fused form casts the small
*matrices* down instead and lets `addmm` fold `scaling` and the residual add.

lora.py's own docstring names this as the mainstream arrangement -- "every
mainstream LoRA implementation (HF PEFT, diffusers) keeps trainable adapter
weights in fp32 even when the frozen base model is fp16/bf16, casting down
for the forward matmul" -- so it is not an exotic change. But "mainstream" is
not a measurement, so this measures the difference:

  * the relative error of the layer's OUTPUT, which is what the next layer and
    the loss see;
  * and the error in the LoRA DELTA alone, which is the part that carries the
    adapter's contribution and is the quantity a numerics claim should be about
    -- a small relative error on the delta can still be a large relative error
    on a LoRA whose whole job is a small change to a large output.

Reported for the LoRA delta at several ranks and at a few scales of adapter
magnitude, because the error depends on both: rank sets how many terms are
summed, and lora_B starts at zero so the delta's size relative to the base
output is what varies over a run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from nodes.model.lora import LoRALinear


def current_forward(mod, x):
    """Verbatim from nodes/model/lora.py's LoRALinear.forward (gate unset)."""
    result = F.linear(x, mod.base_weight, mod.base_bias)
    lora_out = (mod.dropout(x).to(mod.lora_A.dtype) @ mod.lora_A.T) @ (
        mod.lora_B.T * mod.scaling)
    return result + lora_out.to(result.dtype)


def fused_forward(mod, x):
    """The candidate: A/B cast down, addmm folds scaling and the residual add."""
    result = F.linear(x, mod.base_weight, mod.base_bias)
    dtype = result.dtype
    a, b = mod.lora_A.to(dtype), mod.lora_B.to(dtype)
    lead = x.shape[:-1]
    h = x.reshape(-1, x.shape[-1]) @ a.T
    out = torch.addmm(result.reshape(-1, result.shape[-1]), h, b.T,
                      beta=1, alpha=mod.scaling)
    return out.view(*lead, out.shape[-1])


def delta_of(mod, x):
    """The adapter's own contribution: output - base output. Isolated so the
    comparison is about the delta and not swamped by the base weight."""
    base = F.linear(x, mod.base_weight, mod.base_bias)
    return current_forward(mod, x) - base


def rel_err(a, b) -> float:
    denom = a.float().norm().item()
    return float((a.float() - b.float()).norm().item() / denom) if denom else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    print(f"CPU, bf16 base, batch {args.batch} x {args.tokens} tokens; "
          f"lora_A/B fp32 (as they must be for the optimizer)\n")
    print(f"{'rank':>5}{'in/out':>10}{'scaling':>9}"
          f"{'|B|':>9}{'delta rel err':>16}{'output rel err':>17}")
    print("-" * 66)
    worst = 0.0
    for rank in (16, 64, 128):
        for in_f, out_f in ((320, 320), (1280, 1280)):
            for b_scale in (0.01, 0.1):
                base = torch.nn.Linear(in_f, out_f, bias=False,
                                       dtype=torch.bfloat16)
                mod = LoRALinear(base, rank=rank, alpha=32.0)
                with torch.no_grad():
                    mod.lora_B.copy_(torch.randn_like(mod.lora_B) * b_scale)
                x = torch.randn(args.batch, args.tokens, in_f,
                                dtype=torch.bfloat16)
                y_old = current_forward(mod, x)
                y_new = fused_forward(mod, x)
                d_old = delta_of(mod, x)
                d_new = (fused_forward(mod, x)
                         - F.linear(x, mod.base_weight, mod.base_bias))
                e_delta = rel_err(d_old, d_new)
                e_out = rel_err(y_old, y_new)
                worst = max(worst, e_delta)
                print(f"{rank:>5}{f'{in_f}/{out_f}':>10}"
                      f"{mod.scaling:>9.4f}{b_scale:>9.2f}"
                      f"{e_delta:>15.2%}{e_out:>17.2%}")
    print()
    print(f"worst delta relative error {worst:.2%}")
    print()
    print("bf16 carries ~8 mantissa bits, so a single rounding is ~0.4%; a")
    print("rank-64 dot product of two bf16 values accumulates in fp32 and")
    print("rounds once on output, so the delta error is expected to sit near")
    print("that figure rather than grow with rank. The output error is")
    print("necessarily smaller still, because the base weight dominates it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
